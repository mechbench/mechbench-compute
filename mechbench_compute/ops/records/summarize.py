from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.expand_cells import expand_cells
from mechbench_compute.blocks.read_group_key import read_group_key
from mechbench_compute.blocks.read_interval import read_interval
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.reduce.monoid import Monoid

OP = Op(
    name="records/summarize",
    summary=(
        "Group records by coordinates and summarise a numeric field — count, "
        "median, mean, min, max and the share below zero — as a table."
    ),
    description="""\
One row per distinct combination of the `by` coordinates (or one row in all
when `by` is empty). `share_negative` is the fraction of values below zero,
useful when the field is a delta.

A grid — a patch trace, a head sweep, a lens read-out, anything with
`axes` and `measures` — is summarised cell by cell: each cell is a record
with the axes as fields (and `token` beside `position` when the grid
carries tokens) and each measure a column. `value: "share", by:
["position"]` over a trace is one row per token with the most any layer's
patch there recovers.

A record without the value field is refused by name, because a mean over
"the records that happened to have it" is the kind of number nobody
notices is wrong. When absent values are expected — a judge that could not
be read, an unscored item — set `on_missing: "skip"` and the count of
skipped records is reported on the table as `n_missing`.

### How sure

A mean over fifteen prompts is a point; `interval: 0.95` puts an interval
around it — `lo` and `hi`, the percentile bootstrap of the mean over
`resamples` redraws of the group's records under `seed` — so a peak in a
sweep is a claim with a width, not a number. Whether two groups DIFFER is
`records/contrast`'s question, which pairs the records first.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=(
        Output('records/table', collection=False, doc='One row per group with the `by` coordinates and `n`, `median`, `mean`, `min`, `max`, `share_negative`, plus `lo` and `hi` when an `interval` was asked; `n_missing` when any were skipped; the header\'s `interval` says the level, method, resamples and seed.')
    ),
    params=(
        P("value", "string",
          "The numeric field to summarise. A record has many numeric "
          "fields; this names the one the question is about."),
        P("by", "list[string]",
          "The coordinates to group on. Empty gives one overall row.",
          None),
        P("on_missing", "string",
          "`\"error\"`: refuse a record without the field. `\"skip\"`: omit "
          "it and report how many were omitted.",
          "error", choices=("error", "skip")),
        P("interval", "float",
          "The level of a bootstrap interval on each group's mean — `0.95` "
          "adds `lo` and `hi` to every row. None reports the point alone.",
          None),
        P("resamples", "int",
          "How many bootstrap redraws the interval is read from.",
          2000),
    ),
    example={"value": "delta", "by": ["genre", "alpha"], "interval": 0.95},
    example_inputs={"records": {"$ref": {"bench": "you/lab/deltas"}}},
)


def run(ctx, inputs, params):
    return group_stats(inputs["records"], params)


def _bootstrap_mean(values: Sequence[float], level: float, resamples: int,
                    seed: int) -> tuple[float, float]:
    """A percentile bootstrap interval on the mean: the records
    resampled with replacement `resamples` times under `seed`. A
    single value's interval is the value itself."""
    import numpy as np

    # Sorted first: the draw is then a function of the multiset, not of
    # the order the records arrived in — the law every pure block keeps.
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.size < 2:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(int(resamples), v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def summarize_groups(groups: Mapping[tuple, Sequence[float]], params: Mapping[str, Any]) -> dict[str, Any]:
    """The `records/summarize` table from values grouped by the `by`
    key — shared by the flat block and its monoid, so the two are the
    same rows by construction. With `interval`, every row carries the
    bootstrap `lo`/`hi` of its mean."""
    from statistics import median

    by = params.get("by") or []
    value_field = params["value"]
    interval = read_interval(params)
    rows = []
    for key, vals in groups.items():
        vals = list(vals)
        row = {k: key[i] for i, k in enumerate(by)}
        row.update({
            "n": len(vals),
            "median": round(median(vals), 4),
            "mean": round(math.fsum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "share_negative": round(sum(v < 0 for v in vals) / len(vals), 3),
        })
        if interval is not None:
            lo, hi = _bootstrap_mean(vals, *interval)
            row.update({"lo": round(lo, 4), "hi": round(hi, 4)})
        rows.append(row)
    stats = ["n", "median", "mean", "min", "max", "share_negative"]
    if interval is not None:
        stats += ["lo", "hi"]
    columns = [{"name": k, "dtype": "string"} for k in by] + [
        {"name": n, "dtype": "number"} for n in stats]
    out = {"kind": "records/table",
           "name": params.get("name", f"{value_field}-stats"),
           "description": params.get("description", ""),
           "row_axis": "condition", "columns": columns, "rows": rows}
    if interval is not None:
        level, resamples, seed = interval
        out["interval"] = {"level": level, "method": "percentile-bootstrap",
                           "of": "mean", "resamples": resamples, "seed": seed}
    return out


def group_stats(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """Group records by coords and summarize a numeric field into
    MetricTable-shaped rows. by: [coord names] ([] = one overall
    group); value: field name; stats fixed: n/median/mean/min/max +
    share_negative (useful for deltas), and with `interval` the
    bootstrap `lo`/`hi` of the mean.

    `on_missing` says what a record without the value field means:
    `error` (default) refuses by name, because a mean over the records
    that happened to have the field is the kind of number nobody
    notices is wrong; `skip` omits them and REPORTS the count, which is
    what a judged corpus needs — an unreadable verdict is not a zero,
    and the rows that were dropped must be visible."""
    recs = expand_cells(read_items(records))
    by = params.get("by") or []
    value_field = params["value"]
    on_missing = str(params.get("on_missing", "error"))
    if on_missing not in ("error", "skip"):
        raise ValueError(
            f"group-stats on_missing must be 'error' or 'skip', not {on_missing!r}")
    groups: dict[tuple, list[float]] = {}
    n_missing = 0
    for r in recs:
        if value_field not in r or r[value_field] is None:
            if on_missing == "skip":
                n_missing += 1
                continue
            raise ValueError(
                f"group-stats: record {r.get('id')!r} has no {value_field!r} "
                f"field. Set on_missing: 'skip' if absent values are expected "
                f"(a judge that could not be read, an unscored item) — the "
                f"count is then reported on the table.")
        key = read_group_key(r, by)
        groups.setdefault(key, []).append(float(r[value_field]))
    out = summarize_groups(groups, params)
    if n_missing:
        out["n_missing"] = n_missing
    return out


class GroupStats(Monoid):
    """The exact monoid form of `group-stats`: per group, the multiset
    of values (sorted); finalize reproduces the block's rows with
    `fsum` means. Bit-identical to the flat block by construction."""

    def identity(self):
        return {}

    def partial(self, records, params):
        from mechbench_compute.blocks import read_group_key, expand_cells

        by = params.get("by") or []
        f = params["value"]
        groups: dict[tuple, list[float]] = {}
        for r in expand_cells(records):
            key = read_group_key(r, by)
            groups.setdefault(key, []).append(float(r[f]))
        return {k: tuple(sorted(v)) for k, v in groups.items()}

    def merge(self, a, b):
        out = {k: v for k, v in a.items()}
        for k, v in b.items():
            out[k] = tuple(sorted(out.get(k, ()) + v))
        return out

    def finalize(self, p, params):
        pass

        ordered = {key: list(p[key]) for key in sorted(p, key=lambda k: tuple(str(x) for x in k))}
        return summarize_groups(ordered, params)


MONOID = GroupStats
