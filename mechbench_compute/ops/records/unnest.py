from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/unnest",
    summary=(
        "One record per element of a list field — a verdict's votes, a "
        "transcript's messages — each keeping its parent's coordinates."
    ),
    description="""\
Some records carry a list whose elements are the things a question is
about: a pairwise verdict holds its three votes, and whether the option
shown first won is a question about votes, not verdicts. This makes each
element a record of its own, so every operation that reads records reads
them.

Each element becomes a record with its parent's `coords`, plus one more
coordinate (named by `index`) holding the element's place in the list,
and `parent`, the parent's id. Its id is the parent's id and that place,
`"flash-0/2"`. An element that is an object contributes its fields
(an `id`, `coords` or `kind` of its own gives way to the new record's); an
element that is a plain value is stored under `as`. Nothing else of the
parent comes along: a field wanted on the elements belongs in a
coordinate, or in a `records/zip` with the parents afterwards.

An empty list contributes no records. A record without the field is
refused by name; `on_missing: "skip"` passes over it and reports the
count as `n_missing`.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=Output('records/record', collection=True, doc="One record per element, in the parents' order and then the list's: `id`, `coords` (the parent's and the `index` coordinate), `parent`, and the element's fields. The header's `unnested` says the field, how many parents, how many records came out, and `n_missing` when any were skipped."),
    params=(
        P("field", "string", "The list field to unnest, or a dot path to it."),
        P("index", "string",
          "The coordinate that holds each element's place in its list.",
          "index"),
        P("as", "string",
          "The field an element that is a plain value (not an object) is "
          "stored under.",
          "value"),
        P("on_missing", "string",
          "`\"error\"`: refuse a record without the field. `\"skip\"`: pass "
          "over it and report how many were passed over.",
          "error", choices=("error", "skip")),
    ),
    example={"field": "votes", "index": "vote"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/verdicts"}}},
)


def run(ctx, inputs, params):
    return unnest(inputs["records"], params)


def unnest(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """`records/unnest`: one record per element of `field`, in order."""
    field = params["field"]
    index = str(params.get("index", "index"))
    as_field = str(params.get("as", "value"))
    on_missing = str(params.get("on_missing", "error"))
    if on_missing not in ("error", "skip"):
        raise ValueError(f"unnest on_missing must be 'error' or 'skip', not {on_missing!r}")
    out: list[dict[str, Any]] = []
    parents = missing = 0
    for r in read_items(records):
        elements = read_field(r, field)
        if elements is None:
            if on_missing == "skip":
                missing += 1
                continue
            raise ValueError(
                f"unnest: record {r.get('id')!r} has no {field!r} field. Set "
                f"on_missing: 'skip' if absent lists are expected.")
        if not isinstance(elements, list):
            raise TypeError(
                f"unnest: record {r.get('id')!r} field {field!r} is a "
                f"{type(elements).__name__}, not a list")
        coords = dict(r.get("coords") or {})
        if index in coords:
            raise ValueError(
                f"unnest: record {r.get('id')!r} already has a {index!r} "
                f"coordinate; name another with `index`")
        parents += 1
        for i, element in enumerate(elements):
            rec: dict[str, Any] = {}
            if isinstance(element, Mapping):
                rec.update({k: v for k, v in element.items() if k not in ("id", "coords", "kind")})
            else:
                rec[as_field] = element
            rec.update({"id": f"{r.get('id')}/{i}", "coords": {**coords, index: i},
                        "parent": r.get("id")})
            out.append(rec)
    unnested = {"field": field, "parents": parents, "records": len(out)}
    if missing:
        unnested["n_missing"] = missing
    return build_collection(out, unnested=unnested, name=params.get("name"),
                            description=params.get("description"))
