from __future__ import annotations

import fnmatch
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mechbench_compute.blocks.read_record_key import read_record_key
from mechbench_compute.calibration.read_numbers import read_numbers
from mechbench_compute.lexicon._base import DIFFERENCE, Notable
from mechbench_compute.live.choose_baseline import CONTROL, Baseline, is_control
from mechbench_compute.live.read_caveats import read_caveats

K_FLOORS = 1.0
MOST_CHANGES = 10
COUNT = "items"
NOISE = "noise"
SMALL = "small"
MOVED = "moved"


@dataclass(frozen=True)
class Reading:
    item: Mapping[str, Any]
    field: str
    index: int | None
    before: float
    after: float
    floor: float | None
    units: float | None
    distance: float


def read_notable(result: Any, notable: Notable, *, kind: str, operation: str,
                 architecture: str | None = None, baseline: Baseline | None = None,
                 noise: Any = None, k: float = K_FLOORS, machine: str | None = None,
                 pure: bool = False) -> dict[str, Any]:
    items, header = read_items_and_header(result)
    caveats = read_caveats(items, header)
    if baseline is None:
        return {"state": None, "baseline": None, "compared": 0, "changes": [], "caveats": caveats}
    floor = find_floor(noise, architecture, operation, notable.field)
    readings = [r for before, after in pair_items(notable, items, header, baseline)
                for r in read_pair(notable, kind, before, after, floor)]
    found: list[dict[str, Any]] = []
    if not readings:
        found.append({"code": "NOTHING_TO_COMPARE", "field": notable.field})
        return {"state": None, "baseline": baseline.to_dict(), "compared": 0, "changes": [],
                "caveats": found + caveats}
    ranked = sorted(readings, key=lambda r: rank(notable, r, floor, k), reverse=True)
    if floor is None and not pure:
        found.append({"code": "NO_FLOOR", "noise": noise is not None, "architecture": architecture,
                      "operation": operation, "field": notable.field})
    if not pure and baseline.machine and machine and baseline.machine != machine:
        found.append({"code": "OTHER_MACHINE", "baseline_machine": baseline.machine, "machine": machine})
    return {"state": judge(notable, ranked[0], floor, k), "baseline": baseline.to_dict(),
            "compared": len(readings),
            "changes": [read_change(notable, r) for r in ranked[:MOST_CHANGES]],
            "caveats": found + caveats}


def read_items_and_header(result: Any) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    from mechbench_compute.lexicon import kinds as K

    if isinstance(result, Mapping) and K.item_kind_of(result) is not None:
        items: Sequence[Any] = K.items_of(result)
        header = {k: v for k, v in result.items() if k not in ("kind", "item_kind", "key", "items")}
    elif isinstance(result, list):
        items, header = result, {}
    elif isinstance(result, Mapping):
        items, header = [result], {}
    else:
        items, header = [], {}
    return [it for it in items if isinstance(it, Mapping)], header


def pair_items(notable: Notable, items: Sequence[Mapping[str, Any]], header: Mapping[str, Any],
               baseline: Baseline) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if baseline.origin == CONTROL:
        controls = {it.get("id"): it for it in items if is_control(notable, it)}
        return [(controls[it.get("id")], it) for it in items
                if not is_control(notable, it) and it.get("id") in controls]
    before_items, before_header = read_items_and_header(baseline.result)
    if not notable.key:
        return [({**before_header, COUNT: len(before_items)}, {**header, COUNT: len(items)})]
    before = index_items(before_items, notable.key)
    return [(before[k], it) for k, it in index_items(items, notable.key).items() if k in before]


def index_items(items: Sequence[Mapping[str, Any]], key: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    for it in items:
        k = json.dumps([read_record_key(it, f) for f in key], sort_keys=True, default=str)
        if k in seen:
            out.pop(k, None)
            continue
        seen.add(k)
        out[k] = it
    return out


def read_paths(item: Mapping[str, Any], field: str) -> list[tuple[str, str | None, Any]]:
    parts = field.split(".")
    out: list[tuple[str, str | None, Any]] = []

    def walk(node: Any, at: int, path: list[str], name: str | None) -> None:
        if at == len(parts):
            out.append((".".join(path), name, node))
            return
        if not isinstance(node, Mapping):
            return
        if parts[at] == "*":
            for key in sorted(node, key=str):
                walk(node[key], at + 1, [*path, str(key)], str(key) if name is None else name)
        elif parts[at] in node:
            walk(node[parts[at]], at + 1, [*path, parts[at]], name)

    walk(item, 0, [], None)
    return out


def read_pair(notable: Notable, kind: str, before: Mapping[str, Any], after: Mapping[str, Any],
              floor: Mapping[str, Any] | None) -> list[Reading]:
    earlier = {path: value for path, _, value in read_paths(before, notable.field)}
    compared: list[tuple[str, int | None, float, float]] = []
    for path, _, value in read_paths(after, notable.field):
        xs = read_numbers(earlier.get(path)) if path in earlier else None
        ys = read_numbers(value)
        if xs is None or ys is None or len(xs) != len(ys):
            continue
        listed = isinstance(value, (list, tuple))
        compared += [(path, i if listed else None, x, y) for i, (x, y) in enumerate(zip(xs, ys))]
    if not compared:
        return []
    distance = (max(abs(y - x) for *_, x, y in compared) if notable.metric == DIFFERENCE
                else measure_distance(kind, notable.metric, before, after))
    return [Reading(after, path, index, x, y, allow(x, y, floor), count_floors(x, y, floor), distance)
            for path, index, x, y in compared]


def measure_distance(kind: str, name: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    from mechbench_compute import metrics

    _, metric, fn = metrics.resolve(kind, name)
    return float(fn([before, after], metrics.options_of(metric, None))[0][1])


def find_floor(noise: Any, architecture: str | None, operation: str, field: str) -> Mapping[str, Any] | None:
    if noise is None or architecture is None:
        return None
    rows = [r for r in read_items_and_header(noise)[0]
            if r.get("architecture") == architecture and r.get("operation") == operation
            and is_same_field(field, str(r.get("field") or ""))]
    if not rows:
        return None
    return max(rows, key=lambda r: (float(r.get("spread") or 0), float(r.get("relative_spread") or 0)))


def is_same_field(declared: str, recorded: str) -> bool:
    a, b = declared.split("."), recorded.split(".")
    return len(a) == len(b) and all(fnmatch.fnmatchcase(x, y) or fnmatch.fnmatchcase(y, x)
                                    for x, y in zip(a, b))


def allow(x: float, y: float, floor: Mapping[str, Any] | None) -> float | None:
    if floor is None:
        return None
    return max(float(floor.get("spread") or 0),
               float(floor.get("relative_spread") or 0) * max(abs(x), abs(y)))


def count_floors(x: float, y: float, floor: Mapping[str, Any] | None) -> float | None:
    allowed = allow(x, y, floor)
    if allowed is None:
        return None
    gap = abs(y - x)
    if gap == 0:
        return 0.0
    return gap / allowed if allowed > 0 else math.inf


def measure(notable: Notable, r: Reading) -> float:
    return abs(r.after - r.before) if notable.metric == DIFFERENCE else r.distance


def is_past(r: Reading, floor: Mapping[str, Any] | None, k: float) -> bool:
    return floor is None or (r.units or 0.0) > k


def rank(notable: Notable, r: Reading, floor: Mapping[str, Any] | None,
         k: float) -> tuple[bool, bool, float, float]:
    past = is_past(r, floor, k)
    return (past and measure(notable, r) > notable.threshold, past,
            measure(notable, r) if past else (r.units or 0.0), abs(r.after - r.before))


def judge(notable: Notable, r: Reading, floor: Mapping[str, Any] | None, k: float) -> str:
    if not is_past(r, floor, k):
        return NOISE
    return MOVED if measure(notable, r) > notable.threshold else SMALL


def read_change(notable: Notable, r: Reading) -> dict[str, Any]:
    return {
        "key": {f: read_record_key(r.item, f) for f in notable.key},
        "field": r.field,
        "index": r.index,
        "before": r.before,
        "after": r.after,
        "difference": r.after - r.before,
        "metric": notable.metric,
        "distance": measure(notable, r),
        "floors": None if r.units is None or math.isinf(r.units) else r.units,
        "floor": r.floor,
        "threshold": notable.threshold,
    }
