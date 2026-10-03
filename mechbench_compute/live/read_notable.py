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
from mechbench_compute.live.say_number import PLACES, say_number

NONE_YET = "first reading here; nothing to compare yet"
K_FLOORS = 1.0
WITHIN_FLOOR = "this difference is within the floor"
MOST_PLACES = 6
COUNT = "items"
MOVED = "moved"
WITHIN = "within"
BELOW = "below"
SAID_BEFORE = {MOVED: "", WITHIN: "within the floor: ", BELOW: "below the threshold: "}


@dataclass(frozen=True)
class Reading:
    item: Mapping[str, Any]
    name: str | None
    index: int | None
    before: float
    after: float
    units: float | None
    distance: float


def read_notable(result: Any, notable: Notable, *, kind: str, operation: str,
                 architecture: str | None = None, baseline: Baseline | None = None,
                 noise: Any = None, k: float = K_FLOORS, machine: str | None = None,
                 pure: bool = False) -> dict[str, Any]:
    items, header = read_items_and_header(result)
    caveats = read_caveats(items, header)
    if baseline is None:
        return {"line": NONE_YET, "moved": False, "baseline": None, "caveats": caveats}
    floor = find_floor(noise, architecture, operation, notable.field)
    readings = [r for before, after in pair_items(notable, items, header, baseline)
                for r in read_pair(notable, kind, before, after, floor)]
    against = f"against {baseline.label}"
    if not readings:
        return {"line": f"{against}, nothing to compare: no item carries {notable.field} on both sides",
                "moved": False, "baseline": baseline.to_dict(), "caveats": caveats}
    best, state = choose_reading(notable, readings, floor, k)
    measured = say_measure(notable, best, floor, state)
    line = f"{against}, {SAID_BEFORE[state]}{say_clause(notable, best, header)} ({measured})"
    moved = state == MOVED
    found: list[dict[str, str]] = []
    if floor is None and not pure:
        found.append({"code": "NO_FLOOR", "line": say_no_floor(noise, architecture, operation, notable)})
    if state == WITHIN and any(r.after != r.before for r in readings):
        found.append({"code": "WITHIN_FLOOR", "line": WITHIN_FLOOR})
    if not pure and baseline.machine and machine and baseline.machine != machine:
        found.append({"code": "OTHER_MACHINE", "line": (
            f"the baseline was read on {baseline.machine} and this on {machine}: "
            "part of any difference may be the machines'")})
    return {"line": line, "moved": moved, "baseline": baseline.to_dict(), "caveats": found + caveats}


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
    compared: list[tuple[str | None, int | None, float, float]] = []
    for path, name, value in read_paths(after, notable.field):
        xs = read_numbers(earlier.get(path)) if path in earlier else None
        ys = read_numbers(value)
        if xs is None or ys is None or len(xs) != len(ys):
            continue
        listed = isinstance(value, (list, tuple))
        compared += [(name, i if listed else None, x, y) for i, (x, y) in enumerate(zip(xs, ys))]
    if not compared:
        return []
    distance = (max(abs(y - x) for *_, x, y in compared) if notable.metric == DIFFERENCE
                else measure_distance(kind, notable.metric, before, after))
    return [Reading(after, name, index, x, y, count_floors(x, y, floor), distance)
            for name, index, x, y in compared]


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


def count_floors(x: float, y: float, floor: Mapping[str, Any] | None) -> float | None:
    if floor is None:
        return None
    gap = abs(y - x)
    if gap == 0:
        return 0.0
    allowed = max(float(floor.get("spread") or 0),
                  float(floor.get("relative_spread") or 0) * max(abs(x), abs(y)))
    return gap / allowed if allowed > 0 else math.inf


def choose_reading(notable: Notable, readings: Sequence[Reading], floor: Mapping[str, Any] | None,
                   k: float) -> tuple[Reading, str]:
    def measure(r: Reading) -> float:
        return abs(r.after - r.before) if notable.metric == DIFFERENCE else r.distance

    def rank(r: Reading) -> tuple[float, float]:
        return measure(r), abs(r.after - r.before)

    past = [r for r in readings if floor is None or (r.units or 0.0) > k]
    moved = [r for r in past if measure(r) > notable.threshold]
    if moved:
        return max(moved, key=rank), MOVED
    if past:
        return max(past, key=rank), BELOW
    return max(readings, key=lambda r: (r.units or 0.0, abs(r.after - r.before))), WITHIN


def say_clause(notable: Notable, best: Reading, header: Mapping[str, Any]) -> str:
    from mechbench_compute.expr.engine import ExprError, load_engine

    change = say_change(best.before, best.after)
    record = {f: read_rounded(v) if is_number(v) else v for f, v in best.item.items()}
    record.update(name=best.name, index=best.index, before=read_rounded(best.before),
                  after=read_rounded(best.after), delta=read_rounded(best.after - best.before), change=change)
    try:
        said = load_engine().render(notable.line, [record], {}, dict(header)).values
    except ExprError:
        said = []
    return str(said[0]) if said and said[0] is not None else f"{notable.field} {change}"


def say_change(before: float, after: float) -> str:
    if before == after:
        return f"stays at {say_number(after)}"
    a, b = say_apart(before, after)
    return f"{'rises' if after > before else 'falls'} from {a} to {b}"


def say_apart(x: float, y: float) -> tuple[str, str]:
    for places in range(PLACES, MOST_PLACES + 1):
        a, b = say_number(x, places), say_number(y, places)
        if a != b:
            return a, b
    return repr(float(x)), repr(float(y))


def read_rounded(x: float) -> float | int:
    said = say_number(x)
    try:
        return int(said)
    except ValueError:
        return float(said)


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def say_measure(notable: Notable, best: Reading, floor: Mapping[str, Any] | None, state: str) -> str:
    parts = []
    if notable.metric != DIFFERENCE:
        parts.append(f"{notable.metric.replace('-', ' ')} {say_number(best.distance)}")
    if floor is not None:
        parts.append(say_floors(best.units or 0.0))
    if state != WITHIN:
        parts.append(f"{'past' if state == MOVED else 'under'} {say_number(notable.threshold)}")
    return ", ".join(parts)


def say_floors(units: float) -> str:
    if math.isinf(units):
        return "above a floor of 0"
    said = say_number(units)
    return f"{said} floor{'' if said == '1' else 's'}"


def say_no_floor(noise: Any, architecture: str | None, operation: str, notable: Notable) -> str:
    judged = f"moved is judged against the threshold, {say_number(notable.threshold)}"
    if noise is None:
        return f"no noise floor: {judged}"
    return f"the noise floor has no record for {architecture} {operation} {notable.field}: {judged}"
