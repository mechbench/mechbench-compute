from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/relabel",
    summary=(
        "Replace a field's values with the words a reader should see — a "
        "port name with its prose label, `true` with \"global attention\"."
    ),
    description="""\
A value that is right for the graph is often wrong for the page: a port
named `attn`, a condition named `flash`, a `true` from `records/lookup`.
`labels` maps each value to what it should say, and every record whose
field holds one of them gets the label instead.

A key matches a value that is the same text, or whose JSON spelling it is:
the key `"true"` matches the value `true`, and `"12"` matches `12`. A
value no key names is kept as it is, unless `others` is `"error"`, which
refuses it by name — the check that a label was written for every value.

The label is written where the value was read from: a coordinate stays a
coordinate, a field a field. With `as`, it is written as that coordinate
instead, and the original is kept.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=Output('records/record', collection=True, doc='The same records in the same order, each value `labels` names replaced by its label.'),
    params=(
        P("field", "string", "The field whose values are relabelled, or a dot path to it."),
        P("labels", "map[string, json]",
          "Value → label: `{\"attn\": \"its attention only\"}`, "
          "`{\"true\": \"global attention\", \"false\": \"local attention\"}`."),
        P("as", "string",
          "Write the label as this coordinate and keep the original. By "
          "default the label replaces the value where it was.",
          None),
        P("others", "string",
          "`\"keep\"`: a value no key names is kept as it is. `\"error\"`: "
          "it is refused by name.",
          "keep", choices=("keep", "error")),
    ),
    example={"field": "global",
             "labels": {"true": "global attention", "false": "local attention"},
             "as": "attention"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/by-layer"}}},
)


def run(ctx, inputs, params):
    return build_collection(relabel(inputs["records"], params))


def relabel(records: Any, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    field = str(params["field"])
    labels = params.get("labels")
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("records/relabel needs `labels`: {\"value\": \"label\", …}")
    others = str(params.get("others", "keep"))
    if others not in ("keep", "error"):
        raise ValueError(f"records/relabel others must be 'keep' or 'error', not {others!r}")
    as_field = params.get("as")
    out = []
    for r in read_items(records):
        value = read_field(r, field)
        key = _read_label_key(value)
        if key not in labels:
            if others == "error":
                raise ValueError(
                    f"records/relabel: record {r.get('id')!r} has {field}={value!r}, "
                    f"which `labels` does not name; it names {sorted(labels)}")
            label = value
        else:
            label = labels[key]
        rec = dict(r)
        if as_field:
            rec["coords"] = {**(r.get("coords") or {}), str(as_field): label}
        elif key in labels:
            _write_back(rec, field, label)
        out.append(rec)
    return out


def _read_label_key(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _write_back(rec: dict[str, Any], field: str, label: Any) -> None:
    coords = rec.get("coords")
    if isinstance(coords, Mapping) and field in coords:
        rec["coords"] = {**coords, field: label}
        return
    if field in rec or "." not in field:
        rec[field] = label
        return
    parts = field.split(".")
    cur = rec
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, Mapping):
            raise ValueError(f"records/relabel writes through objects only, not {field!r}")
        cur[p] = dict(nxt)
        cur = cur[p]
    cur[parts[-1]] = label
