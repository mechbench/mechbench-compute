from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.match_where import match_where, parse_where
from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

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

`where` may instead be a list of conditions in the grammar the API's item
query reads, `PATH OP VALUE` with OP one of `=` `!=` `<` `<=` `>` `>=` `~`:
`["metadata.call.usage.output_tokens>250", "coords.prompt=flash"]`. The
value is JSON when it parses as JSON and text otherwise. A missing field is
null, so `x=null` keeps the records without it and `x!=null` the records
with it. Ordering holds between two numbers or two texts and nothing else;
`~` is text containing the value, case-insensitively, or a list holding it.
A record passes when every condition holds. A condition may also be written
`{"path": "depth", "op": ">=", "value": {"$param": "allowance"}}`, which lets
its value come from a param.

A field name with dots in it is a path from the record's root:
`metadata.coords.prompt` reads a coordinate a corpus keeps in its
metadata, `metadata.call.usage.output_tokens` a provider's count. A name
that is a coordinate or a top-level field is read as that first.

`fields` projects the survivors down to `id`, `coords` and the named fields,
each under the name as written.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=Output('records/record', collection=True, doc='The matching records.'),
    params=(
        P("where", "map[string, string | float | bool | list[string | float | bool]] | list[string | object]",
          "Field → value or list of values: `{\"genre\": \"noir\", "
          "\"leak\": 0}` keeps noir records with no leak. Or a list of "
          "`PATH OP VALUE` conditions: `[\"lex_words>80\"]`.",
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
    coord) filters too — or a list of `PATH OP VALUE` conditions.
    fields: [names] keeps id+coords plus the named fields."""
    recs = read_items(records)
    where = params.get("where") or {}
    conditions = None if isinstance(where, Mapping) else parse_where(where)
    fields = params.get("fields")
    out = []
    for r in recs:
        if conditions is not None:
            ok = match_where(r, conditions)
        else:
            ok = all(read_field(r, k) in (v if isinstance(v, list) else [v])
                     for k, v in where.items())
        if not ok:
            continue
        if fields:
            out.append({"id": r.get("id"), "coords": dict(r.get("coords") or {}),
                        **{f: read_field(r, f) for f in fields}})
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
