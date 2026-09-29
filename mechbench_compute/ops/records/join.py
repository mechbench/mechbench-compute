from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.raise_expr_error import raise_expr_error
from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.blocks.read_order_by import read_order_by
from mechbench_compute.lexicon._base import In, Op, Output, P, Resume

OP = Op(
    name="records/join",
    resume=Resume("restart"),
    summary=(
        "Match each record on the left with the record on the right that "
        "has the same key, and carry it along under a name."
    ),
    description="""\
`on` is an expression evaluated on both sides (the language is in
mechbench-expr's SPEC.md), `coords.fact` or `[coords.prompt, coords.sample]`;
`on_right` gives the right side its own when the key is spelled differently
there. Each left record is kept with its right match under the field named
`as`, so no field of either side is overwritten and later expressions read
it by name:

```
join:   on: coords.fact, as: flags
filter: where: flags.lied_at_34
```

The right side's keys must be unique: two right records with one key is an
error naming the key, since which one to carry would be a guess. With
`how: inner` (the default) a left record with no match is dropped, and the
header's `unmatched` counts them; with `how: left` it is kept with `None`
under `as`. A key that is `None` matches nothing. The left side's
`order_by` (a sort's) is kept.
""",
    inputs=(
        In("left", "collection | records/table",
           "The records that are kept, each with its match.", many=True),
        In("right", "collection | records/table",
           "The records matched to them, one per key.", many=True),
    ),
    output=Output("records/record", collection=True,
                  doc="The left records, in their order, each with its right match under `as`; "
                      "the header's `unmatched` counts the left records with none."),
    params=(
        P("on", "expression", "The key, evaluated on both sides.",
          reads=("left", "right")),
        P("on_right", "expression", "The right side's key, when it is spelled differently there.", None,
          reads=("right",), replaces="on"),
        P("as", "string", "The field the right record is carried under.", "right"),
        P("how", "string", "`inner` drops a left record with no match; `left` keeps it with None.",
          "inner", choices=("inner", "left")),
    ),
    example={"on": "coords.fact", "as": "flags"},
    example_inputs={"left": {"$ref": {"bench": "you/lab/results/j_1/funnel"}},
                    "right": {"$ref": {"bench": "you/lab/results/j_1/flags"}}},
)


def run(ctx, inputs, params):
    return join_records(inputs["left"], inputs["right"], params, getattr(ctx, "run_params", None) or {})


def join_records(left: Any, right: Any, params: Mapping[str, Any], run_params: Mapping[str, Any]) -> dict[str, Any]:
    import json

    from mechbench_compute.expr.engine import ExprError, load_engine
    from mechbench_compute.lexicon import kinds as K

    lefts, rights = read_items(left), read_items(right)
    on = str(params["on"])
    on_right = str(params.get("on_right") or on)
    as_field = str(params.get("as") or "right")
    how = str(params.get("how", "inner"))
    if how not in ("inner", "left"):
        raise ValueError(f"records/join: how is inner or left, not {how!r}")
    engine = load_engine()
    try:
        left_keys = engine.evaluate(on, lefts, run_params, read_header(left)).values
    except ExprError as e:
        raise_expr_error("records/join", e, lefts)
    try:
        right_keys = engine.evaluate(on_right, rights, run_params, read_header(right)).values
    except ExprError as e:
        raise_expr_error("records/join", e, rights)
    by_key: dict[str, Mapping[str, Any]] = {}
    for record, key in zip(rights, right_keys, strict=True):
        if key is None:
            continue
        k = json.dumps(key, sort_keys=True)
        if k in by_key:
            raise ValueError(
                f"records/join: two right records have the key {key!r} "
                f"({by_key[k].get('id')!r} and {record.get('id')!r}); the right side's keys must be unique")
        by_key[k] = record
    out = []
    unmatched = 0
    for record, key in zip(lefts, left_keys, strict=True):
        match = None if key is None else by_key.get(json.dumps(key, sort_keys=True))
        if match is None:
            unmatched += 1
            if how == "inner":
                continue
        out.append({**record, as_field: match})
    return K.collection("records/record", out, unmatched=unmatched, order_by=read_order_by(left))
