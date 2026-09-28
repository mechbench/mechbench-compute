from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.aggregate_values import AGGREGATES, OBJECT_AGGREGATES, aggregate_values
from mechbench_compute.blocks.raise_expr_error import raise_expr_error
from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.blocks.set_field import set_field
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/group",
    summary=(
        "Gather the records that share a key and compute aggregates over each "
        "group: counts, means, rates with intervals, correlations, paired differences."
    ),
    description="""\
`by` names the group's fields and the expression each is read from (the
language is in mechbench-expr's SPEC.md); records whose values agree are
one group, and a group of none is the whole input: one record, even
when the input is empty (`count()` is then 0). `aggregates` names each
output field and the aggregate call that computes it; the call's positional
arguments are expressions read per record, its named ones settings read
once:

```
by:
  coords.genre: coords.genre
aggregates:
  n: count()
  mean_p: mean(p)
  lied: wilson(top[0].token.text != tracked.truth.token, level=0.9)
```

The plain aggregates are `count()`, `count(cond)`, `sum(x)`, `mean(x)`,
`median(x)`, `min(x)`, `max(x)`, `share(cond)`, `any(cond)`, `all(cond)`,
`first(x)`, `last(x)` and `collect(x)`. The named methods answer an object:
`wilson(cond, level=0.95)` is `{k, n, rate, lo, hi}`, the Wilson score
interval; `bootstrap_mean(x, level=0.95, resamples=2000, seed=0)` is
`{n, mean, lo, hi}`, each group drawn from its own generator at `seed`;
`spearman(x, y, level=None)` is `{n, rho, lo, hi}`, the interval Fisher's;
`paired_difference(x, on, a, b, paired, level=0.95, resamples=2000, seed=0)`
is `{n, mean_a, mean_b, diff, lo, hi, share_positive}`, the records where
`on == a` against those where `on == b`, matched by `paired` when it is
given, the groups drawn in turn from one generator in the order of their
keys as text. Numbers are not rounded; round them in a derive.

A name that lists several, comma-separated, unpacks a named method's
object into those fields, flat:

```
aggregates:
  rate, lo, hi: wilson(lied)
  n, mean: bootstrap_mean(p)
```

Each listed name must be one of the object's fields (`rate`, `lo` and
`hi` of `wilson`'s `{k, n, rate, lo, hi}`); the others are left out.

A value that is `None` is skipped by every aggregate and counted in the
header's `missing`, by aggregate; `on_missing: fail` refuses it instead.
A group's id is its `by` values joined by `|` (`noir|flash`), or `all`
when there is no `by`, so the stored groups are in the order of their
keys; where two groups' values would join to one id, each id is its
values as a JSON list instead.
""",
    inputs=(In("records", "collection | records/table",
               "The records to group: any collection whose items are records, or a table's rows.",
               many=True),),
    output=Output("records/record", collection=True,
                  doc="One record per group, with the `by` fields and the aggregates; "
                      "the header's `missing` counts the values skipped, by aggregate, when there were any."),
    params=(
        P("by", "map[string, expression]",
          "The group's fields, by name (a dot path sets a nested field), each read from the record by an expression.",
          None),
        P("aggregates", "map[string, expression]",
          "The fields computed over each group, by name, each an aggregate call."),
        P("on_missing", "string",
          "`skip` passes over a value that is None and counts it; `fail` refuses it.",
          "skip", choices=("skip", "fail")),
    ),
    example={"by": {"coords.genre": "coords.genre"},
             "aggregates": {"n": "count()", "lied": "wilson(lied, level=0.9)"}},
    example_inputs={"records": {"$ref": {"bench": "you/lab/results/j_1/funnel"}}},
)


def run(ctx, inputs, params):
    return group_records(inputs["records"], params, getattr(ctx, "run_params", None) or {})


def group_records(records: Any, params: Mapping[str, Any], run_params: Mapping[str, Any]) -> dict[str, Any]:
    from mechbench_compute.expr.engine import ExprError, load_engine
    from mechbench_compute.lexicon import kinds as K

    items = read_items(records)
    header = read_header(records)
    by: Mapping[str, str] = params.get("by") or {}
    aggregates: Mapping[str, str] = params.get("aggregates") or {}
    on_missing = str(params.get("on_missing", "skip"))
    if not aggregates:
        raise ValueError("records/group: give at least one aggregate")
    if on_missing not in ("skip", "fail"):
        raise ValueError(f"records/group: on_missing is skip or fail, not {on_missing!r}")
    engine = load_engine()
    calls: dict[str, dict[str, Any]] = {}
    reads: dict[str, str] = {}
    for name, src in aggregates.items():
        try:
            call = engine.split(src, run_params)
        except ExprError as e:
            raise_expr_error("records/group", e, items)
        if call["function"] not in AGGREGATES:
            raise ValueError(
                f"records/group: {call['function']!r} is not an aggregate "
                f"(they are {', '.join(sorted(AGGREGATES))}): `{src}`")
        low, high = AGGREGATES[call["function"]]
        if not low <= len(call["args"]) <= high:
            takes = (f"{low}" if low == high else f"{low} to {high}") + (" expression" if high == 1 else " expressions")
            raise ValueError(f"records/group: {name}: {call['function']} takes {takes}: `{src}`")
        if "," in name and call["function"] not in OBJECT_AGGREGATES:
            raise ValueError(
                f"records/group: {name}: only {', '.join(sorted(OBJECT_AGGREGATES))} answer an object "
                f"to unpack into several fields, not {call['function']}: `{src}`")
        calls[name] = call
        for i, arg in enumerate(_read_args(call)):
            reads[f"{name}#{i}"] = arg
    try:
        keys = engine.evaluate(dict(by), items, run_params, header).values if by else [{} for _ in items]
        values = engine.evaluate(reads, items, run_params, header).values if reads else [{} for _ in items]
    except ExprError as e:
        raise_expr_error("records/group", e, items)
    groups: dict[str, list[int]] = {} if by else {"[]": []}
    key_of: dict[str, dict[str, Any]] = {} if by else {"[]": {}}
    for i, key in enumerate(keys):
        k = json.dumps([key[n] for n in by], sort_keys=True)
        groups.setdefault(k, []).append(i)
        key_of.setdefault(k, key)
    order = sorted(groups, key=lambda k: tuple(str(key_of[k][n]) for n in by))
    missing: dict[str, int] = {}
    computed: dict[str, dict[str, Any]] = {k: {} for k in groups}
    for name, call in calls.items():
        n_args = len(_read_args(call))
        per_group = {k: [tuple(values[i][f"{name}#{j}"] for j in range(n_args)) for i in groups[k]]
                     for k in groups}
        got, skipped = aggregate_values(name, call, per_group, order, on_missing)
        for k, v in got.items():
            computed[k][name] = v
        if skipped:
            missing[name] = skipped
    ids = {k: "|".join(_read_id_part(key_of[k][n]) for n in by) or "all" for k in groups}
    if len(set(ids.values())) < len(ids):
        ids = {k: k for k in groups}
    out = []
    for k in groups:
        row: dict[str, Any] = {"id": ids[k]}
        for name, value in key_of[k].items():
            row = set_field(row, name, value)
        for name, value in computed[k].items():
            for field, v in _unpack(name, value).items():
                row = set_field(row, field, v)
        out.append(row)
    extra = {"missing": missing} if missing else {}
    return K.collection("records/record", out, **extra)


def _unpack(name: str, value: Any) -> dict[str, Any]:
    if "," not in name:
        return {name: value}
    names = [n.strip() for n in name.split(",")]
    if value is None:
        return dict.fromkeys(names)
    absent = [n for n in names if n not in value]
    if absent:
        raise ValueError(f"records/group: {name}: the answer has no {', '.join(absent)} (it has {', '.join(value)})")
    return {n: value[n] for n in names}


def _read_id_part(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _read_args(call: Mapping[str, Any]) -> Sequence[str]:
    if call["function"] == "paired_difference":
        x, on, a, b, *paired = call["args"]
        return [x, f"({on}) == ({a})", f"({on}) == ({b})", *paired]
    return call["args"]
