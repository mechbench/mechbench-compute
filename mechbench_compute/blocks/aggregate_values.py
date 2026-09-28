from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.compute_spearman import compute_spearman
from mechbench_compute.blocks.estimate_bootstrap_mean import estimate_bootstrap_mean
from mechbench_compute.blocks.estimate_fisher_interval import estimate_fisher_interval
from mechbench_compute.blocks.estimate_paired_difference import estimate_paired_difference
from mechbench_compute.blocks.estimate_wilson import estimate_wilson

AGGREGATES: dict[str, tuple[int, int]] = {
    "count": (0, 1), "sum": (1, 1), "mean": (1, 1), "median": (1, 1), "min": (1, 1), "max": (1, 1),
    "share": (1, 1), "any": (1, 1), "all": (1, 1), "first": (1, 1), "last": (1, 1), "collect": (1, 1),
    "wilson": (1, 1), "bootstrap_mean": (1, 1), "spearman": (2, 2), "paired_difference": (4, 5),
}
OBJECT_AGGREGATES = frozenset({"wilson", "bootstrap_mean", "spearman", "paired_difference"})
SETTINGS: dict[str, dict[str, Any]] = {
    "wilson": {"level": 0.95},
    "bootstrap_mean": {"level": 0.95, "resamples": 2000, "seed": 0},
    "spearman": {"level": None},
    "paired_difference": {"level": 0.95, "resamples": 2000, "seed": 0},
}


def aggregate_values(
    name: str,
    call: Mapping[str, Any],
    per_group: Mapping[str, Sequence[tuple[Any, ...]]],
    order: Sequence[str],
    on_missing: str,
) -> tuple[dict[str, Any], int]:
    import numpy as np

    function = call["function"]
    settings = dict(SETTINGS.get(function, {}))
    unknown = sorted(set(call["named"]) - set(settings))
    if unknown:
        raise ValueError(f"records/group: {name}: {function} takes no {', '.join(unknown)}: `{call['canonical']}`")
    settings.update(call["named"])
    rng = np.random.default_rng(int(settings["seed"])) if function == "paired_difference" else None
    skipped = 0
    out: dict[str, Any] = {}
    for k in order:
        rows = per_group[k]
        kept = [r for r in rows if all(v is not None for v in r)]
        dropped = len(rows) - len(kept)
        if dropped and on_missing == "fail":
            raise ValueError(f"records/group: {name}: a value is None in {dropped} records: `{call['canonical']}`")
        skipped += dropped
        out[k] = _aggregate(name, function, call["canonical"], kept, settings, rng)
    return out, skipped


def _aggregate(name: str, function: str, src: str, rows: list[tuple[Any, ...]],
               settings: Mapping[str, Any], rng: Any) -> Any:
    xs = [r[0] for r in rows if r]
    if function == "count":
        return sum(_as_bool(name, src, x) for x in xs) if xs else len(rows)
    if function in ("share", "any", "all", "wilson"):
        bs = [_as_bool(name, src, x) for x in xs]
        if function == "share":
            return sum(bs) / len(bs) if bs else None
        if function == "any":
            return any(bs)
        if function == "all":
            return all(bs)
        k, n = sum(bs), len(bs)
        if not n:
            return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
        lo, hi = estimate_wilson(k, n, float(settings["level"]))
        return {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi}
    if function == "first":
        return xs[0] if xs else None
    if function == "last":
        return xs[-1] if xs else None
    if function == "collect":
        return list(xs)
    if function in ("min", "max"):
        if not xs:
            return None
        try:
            return min(xs) if function == "min" else max(xs)
        except TypeError:
            raise ValueError(f"records/group: {name}: cannot order the values: `{src}`") from None
    if function == "spearman":
        ps = [(_as_number(name, src, x), _as_number(name, src, y)) for x, y in rows]
        rho = compute_spearman([x for x, _ in ps], [y for _, y in ps])
        lo = hi = None
        if settings["level"] is not None:
            lo, hi = estimate_fisher_interval(rho, len(ps), float(settings["level"]))
        return {"n": len(ps), "rho": rho, "lo": lo, "hi": hi}
    if function == "paired_difference":
        paired = len(rows[0]) == 4 if rows else False
        a_items = [(r[3] if paired else None, _as_number(name, src, r[0])) for r in rows if r[1] is True]
        b_items = [(r[3] if paired else None, _as_number(name, src, r[0])) for r in rows if r[2] is True]
        if not a_items or not b_items:
            return None
        return estimate_paired_difference(a_items, b_items, paired, rng, int(settings["resamples"]),
                                          float(settings["level"]))
    ns = [_as_number(name, src, x) for x in xs]
    if function == "sum":
        return sum(ns) if all(isinstance(x, int) for x in ns) else math.fsum(ns)
    if not ns:
        return {"n": 0, "mean": None, "lo": None, "hi": None} if function == "bootstrap_mean" else None
    if function == "mean":
        return math.fsum(ns) / len(ns)
    if function == "median":
        from statistics import median
        return median(ns)
    lo, hi = estimate_bootstrap_mean(ns, float(settings["level"]), int(settings["resamples"]), int(settings["seed"]))
    return {"n": len(ns), "mean": math.fsum(ns) / len(ns), "lo": lo, "hi": hi}


def _as_bool(name: str, src: str, x: Any) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"records/group: {name}: a condition is {type(x).__name__}, not a boolean: `{src}`")
    return x


def _as_number(name: str, src: str, x: Any) -> int | float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"records/group: {name}: a value is {type(x).__name__}, not a number: `{src}`")
    return x
