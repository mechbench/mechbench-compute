from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.read_group_key import read_group_key
from mechbench_compute.blocks.read_interval import read_interval
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/contrast",
    summary=(
        "The difference between two conditions' means of a field — a minus "
        "b on one coordinate — with a paired bootstrap interval and how "
        "often the sign held, one row per combination of the other "
        "coordinates."
    ),
    description="""\
`records/summarize` says what each condition's mean is; this says whether
two of them differ. Name the coordinate (`on`) and its two values (`a`,
`b`); every record on either side contributes its `value`, and the row
reports `mean_a`, `mean_b`, their difference `diff`, the interval `lo`–`hi`
at the given level, and `share_positive` — the fraction of bootstrap
redraws in which the difference was above zero, which is how often the
sign held.

**Pair the records.** In a sweep the same prompts run under every
condition, and a prompt that is hard is hard on both sides. `paired`
names the field the two sides share — a prompt's `id`, usually — and the
bootstrap then redraws PAIRS, so each record's own noise cancels the way
it does in the data; only records present on both sides count, and `n`
is their number. Without `paired`, the sides are redrawn independently,
which is right when the two conditions ran on different records and
wider than it needs to be when they did not.

`by` names further coordinates to hold fixed: one row per combination
of them, each its own contrast.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=(
        Output('records/table', collection=False, doc='One row per combination of the `by` coordinates: `on`, `a`, `b`, `n`, `mean_a`, `mean_b`, `diff`, `lo`, `hi`, `share_positive`. The header\'s `interval` says the level, method, whether it was paired, resamples and seed.')
    ),
    params=(
        P("value", "string", "The numeric field compared."),
        P("on", "string", "The coordinate (or top-level field) the two conditions differ on."),
        P("a", "json", "The value of `on` on the first side; the difference is a minus b."),
        P("b", "json", "The value of `on` on the second side."),
        P("paired", "string",
          "The field the two sides share, to redraw pairs — `\"id\"` when "
          "the same records ran under both conditions. None redraws the "
          "sides independently.",
          None),
        P("by", "list[string]",
          "Coordinates to hold fixed: one row per combination.",
          None),
        P("interval", "float", "The level of the bootstrap interval.", 0.95),
        P("resamples", "int", "How many bootstrap redraws the interval is read from.", 2000),
    ),
    example={"value": "delta_logp", "on": "layer", "a": 23, "b": 22, "paired": "id"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/ablation"}}},
)


def run(ctx, inputs, params):
    return contrast(inputs["records"], params)


def _read_field(record: Mapping[str, Any], name: str) -> Any:
    """A field read the way `read_group_key` reads one: coords first, then
    the record itself."""
    coords = record.get("coords") or {}
    return coords[name] if name in coords else record.get(name)


def contrast(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """`records/contrast`: the difference between two conditions'
    means of a field — `a` minus `b` on the coordinate `on`
    — with a bootstrap interval, one row per combination of the other
    `by` coordinates.

    `paired` names the field the two conditions share (a prompt's `id`,
    usually): the bootstrap then resamples PAIRS, so a record's own
    noise cancels the way it does in the data, and only records present
    on both sides count. Without it the two sides are resampled
    independently. `share_positive` is the fraction of resamples whose
    difference is above zero — how often the sign held."""
    import numpy as np

    recs = read_items(records)
    value_field = params["value"]
    on = params["on"]
    a_val, b_val = params["a"], params["b"]
    by = [k for k in (params.get("by") or []) if k != on]
    paired = params.get("paired")
    level, resamples, seed = read_interval({"interval": params.get("interval", 0.95),
                                            "resamples": params.get("resamples", 2000),
                                            "seed": params.get("seed", 0)})
    sides: dict[tuple, dict[str, list[tuple[Any, float]]]] = {}
    for r in recs:
        side = _read_field(r, on)
        if side == a_val:
            which = "a"
        elif side == b_val:
            which = "b"
        else:
            continue
        if value_field not in r or r[value_field] is None:
            raise ValueError(f"contrast: record {r.get('id')!r} has no {value_field!r} field")
        key = read_group_key(r, by)
        pair_key = _read_field(r, paired) if paired else None
        sides.setdefault(key, {"a": [], "b": []})[which].append((pair_key, float(r[value_field])))
    rows = []
    rng = np.random.default_rng(seed)
    for key in sorted(sides, key=lambda k: tuple(str(x) for x in k)):
        a_items, b_items = sides[key]["a"], sides[key]["b"]
        if not a_items or not b_items:
            raise ValueError(
                f"contrast: {on}={a_val!r} and {on}={b_val!r} must both be present"
                + (f" at {dict(zip(by, key))}" if by else ""))
        if paired:
            a_by = {k: v for k, v in a_items}
            b_by = {k: v for k, v in b_items}
            keys = sorted((k for k in a_by if k in b_by), key=str)
            if not keys:
                raise ValueError(
                    f"contrast: no record's {paired!r} appears on both sides"
                    + (f" at {dict(zip(by, key))}" if by else ""))
            a = np.array([a_by[k] for k in keys]); b = np.array([b_by[k] for k in keys])
            n = len(keys)
            diffs = a - b
            point = float(diffs.mean())
            if n >= 2:
                idx = rng.integers(0, n, size=(resamples, n))
                boots = diffs[idx].mean(axis=1)
            else:
                boots = np.full(resamples, point)
        else:
            a = np.sort([v for _, v in a_items]); b = np.sort([v for _, v in b_items])
            n = min(len(a), len(b))
            point = float(a.mean() - b.mean())
            if len(a) >= 2 and len(b) >= 2:
                ia = rng.integers(0, len(a), size=(resamples, len(a)))
                ib = rng.integers(0, len(b), size=(resamples, len(b)))
                boots = a[ia].mean(axis=1) - b[ib].mean(axis=1)
            else:
                boots = np.full(resamples, point)
        lo, hi = np.percentile(boots, [50 * (1 - level), 50 * (1 + level)])
        row = {k: key[i] for i, k in enumerate(by)}
        row.update({
            "on": on, "a": a_val, "b": b_val, "n": int(n),
            "mean_a": round(float(a.mean()), 4), "mean_b": round(float(b.mean()), 4),
            "diff": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "share_positive": round(float((boots > 0).mean()), 3),
        })
        rows.append(row)
    columns = ([{"name": k, "dtype": "string"} for k in by]
               + [{"name": "on", "dtype": "string"}, {"name": "a", "dtype": "string"},
                  {"name": "b", "dtype": "string"}]
               + [{"name": k, "dtype": "number"}
                  for k in ("n", "mean_a", "mean_b", "diff", "lo", "hi", "share_positive")])
    out = {"kind": "records/table",
           "name": params.get("name", f"{value_field}-contrast"),
           "description": params.get("description", ""),
           "row_axis": "condition", "columns": columns, "rows": rows,
           "interval": {"level": level, "method": "percentile-bootstrap",
                        "of": "difference of means", "paired": paired,
                        "resamples": resamples, "seed": seed}}
    return out
