from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.raise_expr_error import raise_expr_error
from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.blocks.set_field import set_field
from mechbench_compute.lexicon._base import In, Op, Output, P, Resume

OP = Op(
    name="records/sort",
    resume=Resume("restart"),
    summary="Order the records by expressions, and keep the first few.",
    description="""\
`by` is a list of expressions, compared in turn: `["coords.prompt", "-p"]`
orders by prompt, then by `p` descending within a prompt (a `-` before an
expression reverses it; the expression itself may be anything numeric,
text or a list). Records whose key is `None` sort last whichever the
direction, and records with equal keys keep their order, so a sort is
stable. `limit` keeps the first so many: the ten largest effects are
`by: ["-effect"], limit: 10`.

Each record's place, from 1, is written into the field `as` names
(`rank`), and the output declares `order_by: ["rank"]`, so the order is
part of the data: it survives storage (a stored collection is otherwise
sorted by its key) and the operations downstream that keep the records.

The keys share one fuel of 1,000,000 steps across the whole collection, not
one per record: the first record is run alone first, with twice its share,
and if it spends more the node is refused before the others run, with
`EXPRESSION_FUEL`: the estimate, the limit, the records and what to read
instead. A comprehension spends steps on every item of the list it walks,
and a built-in over the list (`sum(xs)`, `max(xs)`, `len(xs)`) two in all.
""",
    inputs=(In("records", "collection | records/table",
               "The records to order: any collection, or a table's rows.",
               many=True),),
    output=Output("records/record", collection=True,
                  doc="The records in the new order, of their kind, the first `limit` of them, each with its "
                      "place in `as`; the header's `order_by` names that field."),
    params=(
        P("by", "list[expression]",
          "The keys to order by, compared in turn; a leading `-` reverses one."),
        P("limit", "int", "Keep only the first so many.", None),
        P("as", "string", "The field each record's place, from 1, is written into.", "rank"),
    ),
    example={"by": ["-effect"], "limit": 10},
    example_inputs={"records": {"$ref": {"bench": "you/lab/results/j_1/effects"}}},
)


def run(ctx, inputs, params):
    return sort_records(inputs["records"], params, getattr(ctx, "run_params", None) or {})


def sort_records(records: Any, params: Mapping[str, Any], run_params: Mapping[str, Any]) -> dict[str, Any]:
    from mechbench_compute.expr.engine import ExprError, load_engine
    from mechbench_compute.lexicon import kinds as K

    items = read_items(records)
    keys = [str(k) for k in (params.get("by") or [])]
    if not keys:
        raise ValueError("records/sort: `by` names at least one key")
    exprs = {f"k{i}": k[1:].strip() if k.startswith("-") else k for i, k in enumerate(keys)}
    descending = [k.startswith("-") for k in keys]
    try:
        got = load_engine().evaluate(exprs, items, run_params, read_header(records))
    except ExprError as e:
        raise_expr_error("records/sort", e, items)
    order = list(range(len(items)))
    for i in reversed(range(len(keys))):
        values = [row[f"k{i}"] for row in got.values]
        present = [j for j in order if values[j] is not None]
        missing = [j for j in order if values[j] is None]
        try:
            sort_keys = {j: _read_key(values[j]) for j in present}
        except TypeError as e:
            raise ValueError(f"records/sort: `{keys[i]}` gives a value that has no order: {e}") from None
        if len({k[0] for k in sort_keys.values()}) > 1:
            raise ValueError(f"records/sort: `{keys[i]}` gives numbers and text together, which have no order between them")
        present.sort(key=lambda j: sort_keys[j], reverse=descending[i])
        order = present + missing
    limit = params.get("limit")
    if limit is not None:
        order = order[: int(limit)]
    place = str(params.get("as") or "rank")
    kind = K.item_kind_of(records) if isinstance(records, Mapping) else None
    ranked = [set_field(dict(items[j]), place, i + 1) for i, j in enumerate(order)]
    return K.collection(kind if kind in K.BY_KIND else "records/record", ranked, order_by=[place])


def _read_key(value: Any) -> Any:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, value)
    if isinstance(value, str):
        return (1, value)
    if isinstance(value, list):
        return (2, [_read_key(v) for v in value])
    raise TypeError(f"{type(value).__name__} is not ordered")
