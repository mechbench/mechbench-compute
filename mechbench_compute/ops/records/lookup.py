from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/lookup",
    summary=(
        "Read each record's value through a list or map its input's header "
        "carries — a grid's index to the component it names, a layer to "
        "whether it attends globally."
    ),
    description="""\
A result says what its records mean in its header, once, rather than on
every record: which layers of the model attend to the whole prompt
(`arch.global_layers`, stamped on every result downstream of a model),
what each place on a grid's axis is (`components` on an attribution,
`layers` on a trace run over some of them). This reads a record's value
through one of those, so the meaning becomes a field the records
operations and a chart can use, and nothing about the model is typed into
the graph.

`in` is a dot path into the header. With `by: "index"` the record's value
is a place in that list, or a key of that map, and the element there is
written: a component index `24` through `components` is `"L23"`. With
`by: "member"` the value is looked for among the list's elements, and
whether it is one is written, `true` or `false`: a layer `23` through
`arch.global_layers` is `true`. Give the words a figure should say with
`records/relabel`.

The header is the `records` input's own, or the `header` input's when
one is wired: an operation that remakes the records, such as
`records/unnest` spreading a grid into its cells, leaves the original's
header behind, so wire the original here.

The result is written as the coordinate `as` (by default the field read,
which it then replaces), so the grouping operations and `records/plot` read
it as they read any other. A record without the field, or with a value the
list has no place for, is refused by name; `on_missing: "null"` writes
null instead.
""",
    inputs=(
        In("records", "collection | records/table",
           "The records to work on: any collection of items — records, "
           "decision reads, vectors, verdicts, tree summaries — since every "
           "item has an id and its fields; a table's rows are read as records.",
           many=True),
        In("header", "collection | records/table",
           "The object whose header holds the list, when it is not "
           "`records` — the result the records were made from.",
           required=False),
    ),
    output=Output('records/record', collection=True, doc="The same records in the same order, each with the coordinate `as` set to what its value reads through the header's list. The header's `lookup` says the field, the path and how it was read."),
    params=(
        P("field", "string", "The field whose value is looked up, or a dot path to it."),
        P("in", "string",
          "The dot path in the header to the list or map to read through: "
          "`arch.global_layers`, `components`, `layers`."),
        P("by", "string",
          "`\"index\"`: the value is a place in the list (counting from 0) "
          "or a key of the map, and the element there is written. "
          "`\"member\"`: whether the value is one of the list's elements.",
          "index", choices=("index", "member")),
        P("as", "string",
          "The coordinate written. By default the field read, which it replaces.",
          None),
        P("on_missing", "string",
          "`\"error\"`: refuse a record the list has nothing for. `\"null\"`: "
          "write null.",
          "error", choices=("error", "null")),
    ),
    example={"field": "layer", "in": "arch.global_layers", "by": "member",
             "as": "global"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/results/j_x/by-layer"}}},
)


def run(ctx, inputs, params):
    records = inputs["records"]
    header = inputs.get("header")
    return lookup(records, records if header is None else header, params)


def lookup(records: Any, header: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """`records/lookup`: each record's `field` read through the list or
    map at `in` in `header`, written as the coordinate `as`."""
    field = str(params["field"])
    path = str(params["in"])
    by = str(params.get("by", "index"))
    as_field = str(params.get("as") or field)
    on_missing = str(params.get("on_missing", "error"))
    if by not in ("index", "member"):
        raise ValueError(f"records/lookup by must be 'index' or 'member', not {by!r}")
    if on_missing not in ("error", "null"):
        raise ValueError(f"records/lookup on_missing must be 'error' or 'null', not {on_missing!r}")
    table = _read_header_path(header, path)
    if by == "member" and not isinstance(table, list):
        raise ValueError(
            f"records/lookup by 'member' reads a list; the header's {path!r} "
            f"is a {type(table).__name__}")
    members = table if by == "member" else None
    out = []
    for r in read_items(records):
        value = read_field(r, field)
        if members is not None:
            found, result = value is not None, value in members
        else:
            found, result = _read_entry(table, value)
        if not found:
            if on_missing == "error":
                raise ValueError(
                    f"records/lookup: record {r.get('id')!r} has "
                    f"{field}={value!r}, which the header's {path!r} has no "
                    f"place for. Set on_missing: 'null' if that is expected.")
            result = None
        rec = {k: v for k, v in r.items() if k != as_field}
        rec["coords"] = {**(r.get("coords") or {}), as_field: result}
        out.append(rec)
    return build_collection(out, lookup={"field": field, "in": path, "by": by, "as": as_field})


def _read_header_path(header: Any, path: str) -> list[Any] | Mapping[str, Any]:
    """The list or map at `path` in a result's header."""
    if not isinstance(header, Mapping):
        raise ValueError(
            "records/lookup reads a header, and its input is a bare list; "
            "wire the result that carries one to `header`")
    value: Any = header
    for part in path.split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
    if not isinstance(value, (list, Mapping)):
        keys = sorted(k for k in header if k not in ("items", "rows"))
        raise ValueError(
            f"records/lookup: the header has no list or map at {path!r}; "
            f"its fields are {keys}")
    return value


def _read_entry(table: list[Any] | Mapping[str, Any], value: Any) -> tuple[bool, Any]:
    """(found, element): a list read at an integer place, a map at a key."""
    if isinstance(table, Mapping):
        key = value if isinstance(value, str) else None if value is None else str(value)
        return (key in table, table.get(key)) if key is not None else (False, None)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(table):
        return False, None
    return True, table[value]
