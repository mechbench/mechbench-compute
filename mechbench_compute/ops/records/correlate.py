from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from mechbench_compute.blocks.read_group_key import read_group_key
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.reduce.monoid import Monoid

OP = Op(
    name="records/correlate",
    summary=(
        "The Spearman rank correlation between two numeric fields of the "
        "same records — do they order the records alike — one row per "
        "combination of the `by` coordinates."
    ),
    description="""\
Two instruments read the same conditions — a pointwise score and a
pairwise preference, a probe's accuracy and a patch's recovery — and the
first question is whether they agree on the order. Name the two fields
(`x`, `y`); each record is one point, and the row reports `n` and `rho`,
Spearman's rank correlation: the Pearson correlation of the two fields'
ranks, ties given their average rank.

Ranks, not values, because two instruments on different scales can agree
on every comparison and still not be linear in each other, and because one
condition far from the rest moves a Pearson correlation more than it moves
the order. `rho` is 1 when the two fields order the records identically,
−1 when they reverse it. It is `null` when either field is constant within
a group or the group has fewer than three records: there is no order to
compare.

### How sure

`interval: 0.95` adds `lo` and `hi`: the Fisher z interval with the
Bonett–Wright standard error for Spearman, √((1 + ρ²/2) / (n − 3)). It is
closed-form, needs no seed, and holds its level at the ten or twenty
conditions a ladder or a sweep has, where a bootstrap of the points would
redraw samples in which one field is constant. It needs four records, and
is `null` below that or when `rho` is ±1. By default there is no interval,
and the header says so.

A record without either field is refused by name; `on_missing: "skip"`
omits it and reports the count as `n_missing`.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=(
        Output('records/table', collection=False, doc='One row per group with the `by` coordinates and `n`, `rho`, plus `lo` and `hi` when an `interval` was asked; `n_missing` when any records were skipped. The header\'s `correlation` names the method and the two fields, and `interval` the level and method when one was asked.')
    ),
    params=(
        P("x", "string", "One numeric field: a coordinate or a top-level field."),
        P("y", "string", "The other numeric field."),
        P("by", "list[string]",
          "The coordinates to group on. Empty gives one overall row.",
          None),
        P("on_missing", "string",
          "`\"error\"`: refuse a record without either field. `\"skip\"`: "
          "omit it and report how many were omitted.",
          "error", choices=("error", "skip")),
        P("interval", "float",
          "The level of a Fisher z interval on `rho`. None reports the "
          "point alone.",
          None),
    ),
    example={"x": "pointwise_gap", "y": "preferred_rate", "by": ["prompt"]},
    example_inputs={"records": {"$ref": {"bench": "you/lab/instruments"}}},
)


def run(ctx, inputs, params):
    return correlate(inputs["records"], params)


def correlate(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """`records/correlate` over a whole collection: the monoid's partial
    of every record, finalized, so the flat and the chunked table are one
    computation."""
    m = MONOID()
    return m.finalize(m.partial(read_items(records), params), params)


def rank_average(values: Sequence[float]) -> list[float]:
    """Each value's rank from 0, tied values sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def compute_spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman's rho, or None when it is undefined."""
    if len(xs) < 3:
        return None
    rx, ry = rank_average(xs), rank_average(ys)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    num = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(math.fsum((a - mx) ** 2 for a in rx) * math.fsum((b - my) ** 2 for b in ry))
    return num / den if den else None


def estimate_fisher_interval(rho: float | None, n: int, level: float) -> tuple[float | None, float | None]:
    """The Fisher z interval on Spearman's rho with the Bonett–Wright
    standard error; (None, None) where it is undefined."""
    if rho is None or n < 4 or abs(rho) >= 1.0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + level / 2)
    se = math.sqrt((1 + rho * rho / 2) / (n - 3))
    centre = math.atanh(rho)
    return math.tanh(centre - z * se), math.tanh(centre + z * se)


def _read_field(record: Mapping[str, Any], name: str) -> Any:
    coords = record.get("coords") or {}
    return coords[name] if name in coords else record.get(name)


class RankPoints(Monoid):
    """Per group, the multiset of (x, y) points (sorted), and the records
    skipped. Ranks need every point, so the partial keeps them all; the
    sort makes the finalized rho a function of the multiset, not of the
    order the records arrived in."""

    def identity(self):
        return ({}, 0)

    def partial(self, records: Sequence[Mapping[str, Any]], params):
        fx, fy = params["x"], params["y"]
        by = params.get("by") or []
        on_missing = str(params.get("on_missing", "error"))
        if on_missing not in ("error", "skip"):
            raise ValueError(f"correlate on_missing must be 'error' or 'skip', not {on_missing!r}")
        groups: dict[tuple, list[tuple[float, float]]] = {}
        missing = 0
        for r in records:
            x, y = _read_field(r, fx), _read_field(r, fy)
            if x is None or y is None:
                if on_missing == "skip":
                    missing += 1
                    continue
                absent = fx if x is None else fy
                raise ValueError(
                    f"correlate: record {r.get('id')!r} has no {absent!r} field. "
                    f"Set on_missing: 'skip' if absent values are expected — the "
                    f"count is then reported on the table.")
            groups.setdefault(read_group_key(r, by), []).append((float(x), float(y)))
        return {k: tuple(sorted(v)) for k, v in groups.items()}, missing

    def merge(self, a, b):
        groups = dict(a[0])
        for key, points in b[0].items():
            groups[key] = tuple(sorted(groups.get(key, ()) + points))
        return groups, a[1] + b[1]

    def finalize(self, p, params):
        groups, missing = p
        by = params.get("by") or []
        level = params.get("interval")
        if level is not None:
            level = float(level)
            if not 0.0 < level < 1.0:
                raise ValueError(f"interval must be between 0 and 1 exclusive, not {level}")
        rows = []
        for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
            points = groups[key]
            rho = compute_spearman([x for x, _ in points], [y for _, y in points])
            row = {name: key[i] for i, name in enumerate(by)}
            row.update({"n": len(points), "rho": None if rho is None else round(rho, 4)})
            if level is not None:
                lo, hi = estimate_fisher_interval(rho, len(points), level)
                row.update({"lo": None if lo is None else round(lo, 4),
                            "hi": None if hi is None else round(hi, 4)})
            rows.append(row)
        stats = ["n", "rho"] + (["lo", "hi"] if level is not None else [])
        columns = ([{"name": name, "dtype": "string"} for name in by]
                   + [{"name": c, "dtype": "number"} for c in stats])
        out = {"kind": "records/table",
               "name": params.get("name", f"{params['x']}-{params['y']}-correlation"),
               "description": params.get("description", ""),
               "row_axis": "condition", "columns": columns, "rows": rows,
               "correlation": {"method": "spearman", "x": params["x"], "y": params["y"]},
               "interval": (None if level is None else
                            {"level": level, "method": "fisher-z-bonett-wright", "of": "rho"})}
        if missing:
            out["n_missing"] = missing
        return out


MONOID = RankPoints
