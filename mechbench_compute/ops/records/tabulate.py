from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import Op, Output, P
from mechbench_compute.lexicon.records import _RECORDS

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
""",
    inputs=(_RECORDS,),
    output=Output('records/table', collection=False, doc='`columns` (`{name, dtype}`) and `rows`.'),
    params=(
        P("row_axis", "string",
          "What one row stands for, recorded on the table for its renderer "
          "— `\"record\"`, `\"condition\"`, `\"layer\"`.",
          "record"),
    ),
    example={"row_axis": "condition", "name": "steering deltas"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/deltas"}}},
)


def run(ctx, inputs, params):
    return tabulate_records(inputs["records"], params)


def tabulate_records(records: Any,
                     params: Mapping[str, Any]) -> dict[str, Any]:
    """Present a record stream as a metric table: coords flatten into
    leading columns, remaining scalar fields follow. The generic
    records -> table presenter (delta tables, group stats, ...)."""
    recs = read_items(records)
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
