from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import Op, Output, P
from mechbench_compute.lexicon.records import _RECORDS

OP = Op(
    name="records/select",
    summary=(
        "Keep the records that match a set of field values, and optionally "
        "keep only some of their fields."
    ),
    description="""\
`where` is a map of field → value (or list of acceptable values). A key is
read from the record's `coords` when it is a coordinate, and from the
record itself otherwise — so a field written by `text/measure` (a pattern hit)
filters as easily as a design coordinate. A record passes when every key
matches.

`fields` projects the survivors down to `id`, `coords` and the named fields.
""",
    inputs=(_RECORDS,),
    output=Output('records/record', collection=True, doc='The matching records.'),
    params=(
        P("where", "map[string, string | float | bool | list[string | float | bool]]",
          "Field → value or list of values. `{\"genre\": \"noir\", "
          "\"leak\": 0}` keeps noir records with no leak.",
          None),
        P("fields", "list[string]",
          "Keep only these fields (plus `id` and `coords`). By default the "
          "whole record is kept.",
          None),
    ),
    example={"where": {"genre": ["noir", "fable"], "leak": 0}},
    example_inputs={"records": {"$ref": {"bench": "you/lab/records"}}},
)


def run(ctx, inputs, params):
    return _select_items(inputs["records"], params)


def select(records: Any, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Filter records by equality; optionally project fields.
    where: {key: value | [values]} — a key is read from `coords` when it
    is a coord, and from the record itself otherwise, so a field that
    `text/measure` `annotate` wrote (a pattern hit is a field, not a
    coord) filters too (task 000368). fields: [names] keeps id+coords
    plus the named fields."""
    recs = read_items(records)
    where: Mapping[str, Any] = params.get("where") or {}
    out = []
    for r in recs:
        coords = r.get("coords", {})
        ok = all(
            (coords.get(k) if k in coords else r.get(k))
            in (v if isinstance(v, list) else [v])
            for k, v in where.items()
        )
        if not ok:
            continue
        fields = params.get("fields")
        if fields:
            out.append({"id": r.get("id"), "coords": dict(coords),
                        **{f: r.get(f) for f in fields}})
        else:
            out.append(r)
    return out


def _select_items(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """A filter keeps the kind; a projection does not. Selecting some of
    a collection's items leaves each item exactly as it was, so a subset
    of adapter deltas is still adapter deltas and still compares by what
    that kind declares. `fields` rewrites the items, and what is left
    may no longer satisfy the kind — that lands as a plain record."""
    from mechbench_compute.lexicon import kinds as K

    items = select(records, params)
    if params.get("fields"):
        return build_collection(items)
    kind = K.item_kind_of(records) if isinstance(records, Mapping) else None
    return K.collection(kind if kind in K.BY_KIND else "records/record", items)
