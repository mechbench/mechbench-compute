from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/rename",
    summary=(
        "Rename fields on every record — the one visible adaptation step "
        "between an op that wrote a field under one name and an op that "
        "reads it under another."
    ),
    description="""\
Every op reads the fields it names: the chat-shaped ops read `system`,
`user` and `prefill`; `eval/score` reads `prediction` and `reference`;
`text/measure` and `eval/judge` read `text`. When a record carries the right
value under another name, this op moves it, and the graph shows the move
rather than hiding it in a parameter.

`fields` maps old name → new name. A name may be a dotted path, so a
value can be moved into or out of `coords` (`{"opening": "coords.opening"}`
makes a measurement a coordinate the grouping ops can read) or lifted from
a document's `metadata.coords`. A record without the old field is left as
it is. Everything not named is kept.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=Output('records/record', collection=True, doc='The same records, with the named fields moved.'),
    params=(
        P("fields", "map[string, string]",
          "Old name → new name, each a field or a dotted path such as "
          "`coords.genre`."),
    ),
    example={"fields": {"question": "user", "opening": "coords.opening"}},
    example_inputs={"records": {"$ref": {"bench": "you/lab/records"}}},
)


def run(ctx, inputs, params):
    return build_collection(rename(inputs["records"], params))


def _pop_path(rec: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Remove the value at a dotted path, copying each container on the
    way so the input record is never mutated. (found, value)."""
    parts = path.split(".")
    cur = rec
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, Mapping):
            return False, None
        nxt = dict(nxt)
        cur[p] = nxt
        cur = nxt
    if parts[-1] not in cur:
        return False, None
    return True, cur.pop(parts[-1])


def _set_path(rec: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = rec
    for p in parts[:-1]:
        nxt = cur.get(p)
        nxt = dict(nxt) if isinstance(nxt, Mapping) else {}
        cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def rename(records: Any, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """records/rename: move fields on every record, `fields: {old: new}`.
    A name may be a dotted path (`coords.opening`, `metadata.coords`), so
    a value moves into or out of a nested object. A record without the
    old field is left as it is; everything not named is kept."""
    fields = params.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("records/rename needs `fields`: {\"old\": \"new\", …}")
    out = []
    for r in read_items(records):
        rec = dict(r)
        for old, new in fields.items():
            found, value = _pop_path(rec, str(old))
            if found:
                _set_path(rec, str(new), value)
        out.append(rec)
    return out
