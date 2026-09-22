from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.coll import _coll
from mechbench_compute.blocks.items import _items
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
    return _coll(template(_items(inputs["records"]), params))


def template(records: list[dict[str, Any]],
             params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Instantiate named string templates against each record's values.
    `{axis-name}` placeholders substitute; everything else is verbatim."""
    templates: Mapping[str, str] = params.get("templates") or {}
    out = []
    for rec in records:
        fields = {}
        for fname, tmpl in templates.items():
            s = str(tmpl)
            # Fixpoint substitution (bounded): a level's text may itself
            # contain placeholders (the Marcus elaborate opening embeds
            # {gender}) — passes repeat while substitutions still fire.
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
