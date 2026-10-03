from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.raise_expr_error import raise_expr_error
from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.blocks.read_order_by import read_order_by
from mechbench_compute.lexicon._base import In, Op, Output, P, Resume

OP = Op(
    name="records/filter",
    resume=Resume("restart"),
    summary="Keep the records a condition holds for.",
    description="""\
`where` is an expression that is `True`, `False` or `None` for each record
(the language is in mechbench-expr's SPEC.md): `coords.prompt == "flash"`,
`layer >= 24 and lied`, `metadata.call.stop_reason == "max_tokens"`. A
record is kept only when its condition is `True`: a false condition and an
unknown one (a missing field compared, `None and True`) both drop it, and
the header's `unknown` counts the ones dropped for being unknown, so a
condition that silently read nothing shows itself.

A condition that is not a boolean (a number, a string) is an error, not a
guess: write `theme == 1`, not `theme`. The kept records keep their
order: an `order_by` on the input (a sort's) is kept.

The condition has one fuel of 1,000,000 steps across the whole collection,
not one per record: the first record is run alone first, with twice its
share, and if it spends more the node is refused before the others run,
with `EXPRESSION_FUEL`: the estimate, the limit, the records and what to
read instead. A comprehension spends steps on every item of the list it
walks, and a built-in over the list (`sum(xs)`, `max(xs)`, `len(xs)`) two
in all.
""",
    inputs=(In("records", "collection | records/table",
               "The records to keep some of: any collection, or a table's rows.",
               many=True),),
    output=Output("records/record", collection=True,
                  doc="The records the condition holds for, in their order, of their kind; "
                      "the header's `dropped` and `unknown` count the rest."),
    params=(
        P("where", "expression", "The condition a record is kept for."),
    ),
    example={"where": "coords.condition == \"lie\" and layer >= 24"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/results/j_1/funnel"}}},
)


def run(ctx, inputs, params):
    return filter_records(inputs["records"], params, getattr(ctx, "run_params", None) or {})


def filter_records(records: Any, params: Mapping[str, Any], run_params: Mapping[str, Any]) -> dict[str, Any]:
    from mechbench_compute.expr.engine import ExprError, load_engine
    from mechbench_compute.lexicon import kinds as K

    items = read_items(records)
    where = str(params["where"])
    try:
        got = load_engine().filter(where, items, run_params, read_header(records))
    except ExprError as e:
        raise_expr_error("records/filter", e, items)
    kept = [items[i] for i in got.kept]
    kind = K.item_kind_of(records) if isinstance(records, Mapping) else None
    return K.collection(
        kind if kind in K.BY_KIND else "records/record", kept,
        dropped=len(items) - len(kept), unknown=got.unknown, order_by=read_order_by(records),
        **({"undefined": got.undefined} if got.undefined else {}))
