from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.rank_guest_records import rank_guest_records
from mechbench_compute.blocks.read_guest_records import read_guest_records
from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.blocks.run_guest_json import run_guest_json
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/jq",
    summary="Reshape records with a jq program, run in the sandbox: the escape hatch for nesting, "
            "unnesting and building new objects.",
    description="""\
`program` is a jq program (gojq 0.12, the jq language) whose input `.`
is the records, as a list; `$header` is the input's header and `$params`
the run's params. Its output is the new records: one list of them, or a
stream of objects, one per record.

```
map(select(.coords.prompt == $params.prompt) | {id, text, words: (.text | split(" ") | length)})
.[] | .votes[] as $v | {id: "\\(.id)/\\($v.vote)", coords, winner: $v.winner}
group_by(.coords.genre) | map({id: .[0].coords.genre, n: length})
```

It runs in the sandbox's shell guest (WebAssembly; no network, an empty
filesystem) in strict mode, so the same program over the same records
gives the same bytes on every machine. The program is a param, so it is
part of the protocol and its hash. A record without an `id` is numbered
when none has one; some with and some without is an error. Stored
records are ordered by id, as every collection is; to keep the order the
program wrote (a `sort_by`, a ranking), name a field in `rank`. A program
that does not parse is refused with jq's own message; a ceiling that
trips (`seconds`, `memory_mb`) is named.

The algebra first: `records/derive`, `filter`, `sort`, `join` and
`group` are checked before a run and drawn field by field; this node is
drawn as code.
""",
    inputs=(In("records", "collection | records/table",
               "The records to reshape: any collection, or a table's rows.",
               many=True),),
    output=Output("records/record", collection=True,
                  doc="The records the program wrote, in its order."),
    params=(
        P("program", "string", "A jq program over the records (`.`), with `$header` and `$params`."),
        P("seconds", "float", "The longest the program may run.", 60.0),
        P("memory_mb", "int", "The most memory the guest may use.", 512),
        P("rank", "string",
          "A field to write each record's place into, from 1, so the order the program wrote is kept "
          "(declared as the collection's `order_by`); without one, records are stored in id order.",
          None),
        P("output_mb", "int", "The most JSON the program may write.", 64),
    ),
    example={"program": "map(select(.p > 0.5))"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/results/j_1/funnel"}}},
)


def run(ctx, inputs, params):
    return reshape_records(inputs["records"], params, getattr(ctx, "run_params", None) or {})


def reshape_records(records: Any, params: Mapping[str, Any], run_params: Mapping[str, Any]) -> dict[str, Any]:
    from mechbench_compute.lexicon import kinds as K

    program = str(params["program"])
    argv = ["jq", "-c", "--argjson", "header", json.dumps(read_header(records), ensure_ascii=False),
            "--argjson", "params", json.dumps(dict(run_params), ensure_ascii=False), program]
    values = run_guest_json("records/jq", "mbshell", argv, read_items(records),
                            memory_mb=int(params.get("memory_mb", 512)),
                            seconds=float(params.get("seconds", 60.0)),
                            output_mb=int(params.get("output_mb", 64)))
    items, header = rank_guest_records(read_guest_records("records/jq", values), params.get("rank"))
    return K.collection("records/record", items, **header)
