from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/tabulate",
    summary=(
        "Present a record list as a table — coordinates become the leading "
        "columns, scalar fields follow."
    ),
    description="""\
The generic records-to-table step. Every coordinate seen across the records
becomes a column, then every scalar (number or string) field; each column's
type is inferred from its values. Nested fields are left out.

A table's rows keep an order, and a collection's items do not: a
collection is stored in the order of its key, whatever order an operation
made it in. So this is where records are put in order, for a chart that
draws its panels, series and bars in the order their values first appear.
`by` names the fields, the first deciding and each later one breaking the
ties of the ones before. A field is ordered by its value — numbers
numerically, text alphabetically — unless `order` lists its values, in
which case a row goes where its value is in the list: `{"removed": ["the
whole layer", "its attention only"]}`. A value the list does not name
comes after every value it does. A row without the field comes last.
`descending` reverses the natural order of the fields `order` does not
list. Rows that tie keep the order they arrived in.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=Output('records/table', collection=False, doc='`columns` (`{name, dtype}`) and `rows`.'),
    params=(
        P("row_axis", "string",
          "What one row stands for, recorded on the table for its renderer "
          "— `\"record\"`, `\"condition\"`, `\"layer\"`.",
          "record"),
        P("by", "string | list[string]",
          "The field to order the rows by, or the fields, the first "
          "deciding; each a field or a dot path. The records' own order "
          "when not given.",
          None),
        P("order", "map[string, list[json]]",
          "Field → its values in the order wanted, for a field whose order "
          "is not its natural one.",
          None),
        P("descending", "bool",
          "Largest first, for the fields `order` does not list.",
          False),
    ),
    example={"row_axis": "condition", "name": "steering deltas",
             "by": ["removed", "layer"],
             "order": {"removed": ["the whole layer", "its attention only"]}},
    example_inputs={"records": {"$ref": {"bench": "you/lab/deltas"}}},
)


def run(ctx, inputs, params):
    return tabulate_records(inputs["records"], params)


def tabulate_records(records: Any,
                     params: Mapping[str, Any]) -> dict[str, Any]:
    """Present a record stream as a metric table: coords flatten into
    leading columns, remaining scalar fields follow. The generic
    records -> table presenter (delta tables, group stats, ...)."""
    recs = sort_records(read_items(records), params)
    coord_keys: list[str] = []
    value_keys: list[str] = []
    for r in recs:
        for k in r.get("coords", {}):
            if k not in coord_keys:
                coord_keys.append(k)
        for k, v in r.items():
            if k in ("id", "coords") or not isinstance(v, (int, float, str)):
                continue
            if k not in value_keys:
                value_keys.append(k)
    rows = []
    for r in recs:
        row: dict[str, Any] = {"id": r.get("id")}
        row.update({k: r.get("coords", {}).get(k) for k in coord_keys})
        row.update({k: r.get(k) for k in value_keys if k in r})
        rows.append(row)
    dtypes = {}
    for k in ["id", *coord_keys, *value_keys]:
        vals = [row.get(k) for row in rows if row.get(k) is not None]
        dtypes[k] = ("number" if vals and all(
            isinstance(v, (int, float)) for v in vals) else "string")
    return {"kind": "records/table",
            "name": params.get("name", "records"),
            "description": params.get("description", ""),
            "row_axis": params.get("row_axis", "record"),
            "columns": [{"name": k, "dtype": d} for k, d in dtypes.items()],
            "rows": rows}


def sort_records(recs: list[Any], params: Mapping[str, Any]) -> list[Any]:
    """The records ordered by `by` and `order`, stably; as they came when
    `by` is not given."""
    by = params.get("by")
    fields = [by] if isinstance(by, str) else list(by or [])
    order = params.get("order") or {}
    if not isinstance(order, Mapping):
        raise ValueError("records/tabulate order is a map: {\"field\": [value, …]}")
    unknown = sorted(set(order) - set(fields))
    if unknown:
        raise ValueError(f"records/tabulate order names {unknown}, which `by` does not")
    descending = bool(params.get("descending", False))
    recs = list(recs)
    # A stable sort by the last field first leaves the first field deciding.
    for field in reversed(fields):
        listed = order.get(field)
        if listed is not None:
            place = {_read_place_key(v): i for i, v in reversed(list(enumerate(listed)))}
            recs.sort(key=lambda r, f=field, p=place: p.get(_read_place_key(read_field(r, f)), len(p)))
        else:
            recs = _sort_naturally(recs, field, descending)
    return recs


def _read_place_key(value: Any) -> str:
    """A value as `order` lists it, compared by its JSON so that `1` and
    `1.0` meet and `true` stays apart from `1`."""
    return json.dumps(float(value) if isinstance(value, int) and not isinstance(value, bool)
                      else value, sort_keys=True)


def _sort_naturally(recs: list[Any], field: str, descending: bool) -> list[Any]:
    """By the field's value, rows without it last whichever the direction."""
    present = [r for r in recs if read_field(r, field) is not None]
    missing = [r for r in recs if read_field(r, field) is None]
    kinds = {"number" if isinstance(v, (int, float)) and not isinstance(v, bool)
             else type(v).__name__ for v in (read_field(r, field) for r in present)}
    if len(kinds) > 1:
        raise ValueError(
            f"records/tabulate: {field!r} holds values of more than one type "
            f"({', '.join(sorted(kinds))}); list the order with `order`")
    present.sort(key=lambda r: read_field(r, field), reverse=descending)
    return present + missing
