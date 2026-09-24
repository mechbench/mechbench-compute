from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/fill",
    summary=(
        "Fill named string templates from each record's factor values — "
        "turn a design into prompts."
    ),
    description="""\
For every record, each entry of `templates` becomes a field of the same
name with `{factor}` placeholders replaced by the record's `values`. A
substituted value may itself contain placeholders (an elaborate opening that
embeds `{gender}`); substitution repeats until nothing changes, up to four
passes. Everything outside braces is verbatim.

The output records keep their `id` and `coords`. Name the templates after
the fields the next op reads — `system`, `user`, `prefill` for the
chat-shaped ops — and no adaptation step is needed between them.
""",
    inputs=(
        In("records", "records/record",
           "Records with `values`, usually from `records/cross`.", many=True),
    ),
    output=Output('records/record', collection=True, doc='One record per input record: `{id, coords}` plus one field per template.'),
    params=(
        P("templates", "map[string, string]",
          "Field name → template string. `{name}` is replaced by the "
          "record's value for factor `name`.",
          None),
    ),
    example={
        "templates": {
            "system": "You are a storyteller.",
            "user": "Write {genre}. Begin with the phrase: {seed}",
        },
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/design"}}},
)


def run(ctx, inputs, params):
    return build_collection(fill_templates(read_items(inputs["records"]), params))


def fill_templates(records: list[dict[str, Any]],
                   params: Mapping[str, Any]) -> list[dict[str, Any]]:
    templates: Mapping[str, str] = params.get("templates") or {}
    out = []
    for rec in records:
        fields = {}
        for fname, tmpl in templates.items():
            s = str(tmpl)
            for _ in range(4):
                before = s
                for axis, value in rec.get("values", {}).items():
                    s = s.replace("{" + axis + "}", str(value))
                if s == before:
                    break
            fields[fname] = s
        out.append({"id": rec["id"], "coords": dict(rec.get("coords", {})),
                    **fields})
    return out
