from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.rank_guest_records import rank_guest_records
from mechbench_compute.blocks.read_guest_records import read_guest_records
from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.blocks.run_guest_json import run_guest_json
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/python",
    summary="Transform records with a Python function, run in the sandbox: the escape hatch for what "
            "the expression algebra does not say.",
    description="""\
`source` is Python that defines `transform`. With `over: records` (the
default) it is called once, `transform(records, header, params)`, and
returns the new records as a list; with `over: record` it is called per
record, `transform(record, header, params)`, and returns a record, a
list of them, or `None` to drop it. `records` are the input's items as
JSON values (dicts and lists), `header` its header, `params` the run's
params.

```python
def transform(records, header, params):
    by = {}
    for r in records:
        by.setdefault(r["coords"]["prompt"], []).append(r["p"])
    return [{"id": k, "p_range": max(v) - min(v)} for k, v in sorted(by.items())]
```

The function runs in the sandbox's CPython 3.13 guest (WebAssembly; no
network, no filesystem but an empty one, pure-Python standard library
only), in strict mode: the clock is virtual and `random` is seeded from
the input, so the same source over the same records gives the same
bytes on every machine. The source is a param, so it is part of the
protocol and its hash. A record the function returns without an `id` is
numbered when none has one; some with and some without is an error.
Stored records are ordered by id, as every collection is; to keep the
order the code wrote (a ranking, a sort), name a field in `rank`. A
failure is refused with the end of the traceback; a ceiling that trips
(`seconds`, `memory_mb`, the fuel every guest runs under) is named.

The algebra first: `records/derive`, `filter`, `sort`, `join` and
`group` are checked before a run and drawn field by field; this node is
drawn as code.
""",
    inputs=(In("records", "collection | records/table",
               "The records to transform: any collection, or a table's rows.",
               many=True),),
    output=Output("records/record", collection=True,
                  doc="The records the function returned, in its order."),
    params=(
        P("source", "string", "Python source that defines `transform`."),
        P("over", "string",
          "`records`: call `transform` once with them all; `record`: once per record.",
          "records", choices=("records", "record")),
        P("seconds", "float", "The longest the function may run.", 60.0),
        P("memory_mb", "int", "The most memory the guest may use.", 512),
        P("rank", "string",
          "A field to write each record's place into, from 1, so the order the code wrote is kept "
          "(declared as the collection's `order_by`); without one, records are stored in id order.",
          None),
        P("output_mb", "int", "The most JSON the function may return.", 64),
    ),
    example={"source": "def transform(records, header, params):\n    return [r for r in records if r['p'] > 0.5]\n"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/results/j_1/funnel"}}},
)

HARNESS = r'''
import json, sys
req = json.load(sys.stdin)
ns = {"__name__": "records_python"}
exec(compile(req["source"], "<source>", "exec"), ns)
fn = ns.get("transform")
if not callable(fn):
    sys.stderr.write("the source defines no function `transform`\n")
    sys.exit(3)
if req["over"] == "record":
    out = []
    for r in req["records"]:
        got = fn(r, req["header"], req["params"])
        if got is None:
            continue
        out.extend(got if isinstance(got, list) else [got])
else:
    out = fn(req["records"], req["header"], req["params"])
if not isinstance(out, list):
    sys.stderr.write("transform must return a list of records, not " + type(out).__name__ + "\n")
    sys.exit(3)
sys.stdout.write(json.dumps(out, allow_nan=False, ensure_ascii=False))
'''


def run(ctx, inputs, params):
    return transform_records(inputs["records"], params, getattr(ctx, "run_params", None) or {})


def transform_records(records: Any, params: Mapping[str, Any], run_params: Mapping[str, Any]) -> dict[str, Any]:
    from mechbench_compute.lexicon import kinds as K

    over = str(params.get("over", "records"))
    if over not in ("records", "record"):
        raise ValueError(f"records/python: over is records or record, not {over!r}")
    request = {"source": str(params["source"]), "over": over, "records": read_items(records),
               "header": read_header(records), "params": dict(run_params)}
    values = run_guest_json("records/python", "cpython", ["python", "-c", HARNESS], request,
                            memory_mb=int(params.get("memory_mb", 512)),
                            seconds=float(params.get("seconds", 60.0)),
                            output_mb=int(params.get("output_mb", 64)))
    items, header = rank_guest_records(read_guest_records("records/python", values), params.get("rank"))
    return K.collection("records/record", items, **header)
