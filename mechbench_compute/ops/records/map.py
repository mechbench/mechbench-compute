from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.api import (
    In,
    Op,
    Output,
    P,
    Resume,
    collection,
    describe_spend_cap,
    item_kind_of,
    items_of,
    read_body_level,
)

OP = Op(
    name="records/map",
    needs=frozenset({"executor.sub"}),
    resume=Resume("restart", items=True),
    summary=(
        "Run a whole sub-protocol once per record — the fan-out between a "
        "node that loops over its own items and a run set that loops over "
        "whole runs."
    ),
    description="""\
A run set fans out over runs; a node fans out over the items inside it.
Between the two there was nothing — so a graph that wanted to do several
things to each record had to be written once per record, or flattened
into one node that knew how to do all of them.

`body` is a graph, run once per record with that record's fields bound
into its params by `bind`: `{"topic": "user"}` puts each record's `user`
field in the body's `{"$param": "topic"}`. The body is written once,
against one record, and reads as what it is. The run's own params reach
the body too, so a body that names `{"$param": "model"}` gets the run's
model; `bind` shadows them.

Each invocation is an **item keyed by the record's id**, so an
interrupted map resumes exactly as an interrupted chat node does: the
records already done are reused, the rest are run. The body sees one
record and nothing else, which is also why chunking a map is the same
map: there is no state between records to lose.

`over` is the other way to give the stream: a list of values, or
`{"range": [0, 42]}`, each becoming a record of one field named by `as`.
`{"over": {"range": [0, 42]}, "as": "layer"}` runs the body once per
layer with `{"$param": "layer"}` filled in — where before the integers
had to be stored as a corpus first.

`collect` says what comes back. `stream` (the default) flattens every
invocation's output into one collection, each item's id prefixed with
its record's and carrying a `mapped` coordinate; `first` keeps one item
per record; `all` keeps each invocation's items nested under its record.
When the body ends at more than one node, `output` names the one to
collect.

`budget_usd` caps the provider spend of every invocation together, so a
loop over a judge has one total. The map stops at the record whose call
would pass it, naming that record; the header carries `spent_usd` and
`budget_usd`. A resumed map counts only what this run spent.
""",
    inputs=(
        In("records", "collection",
           "The stream to map over: one invocation of the body per item, of "
           "any kind — records, or a training's `checkpoints`, one adapter "
           "each. `over` is the other way to give one.",
           many=True, required=False),
    ),
    output=Output('records/record', collection=True, doc="Under `stream`, every invocation's items in one collection, each id prefixed `<record>:<item>` and carrying `coords.mapped`; under `first` or `all`, one item per record. Under `stream` and `first` the collection takes the BODY's item kind — a map over transcripts that produces transcripts emits transcripts — and under `all`, where each item nests a list, it is a plain record. The header's `mapped` says how many records ran, under which policy, and what the body's nodes were; with a `budget_usd`, the header carries it and `spent_usd`, the body's spend over every invocation."),
    params=(
        P("body", "object",
          "The graph to run per record — `{nodes, edges}`, the same shape a "
          "protocol's graph has. Its `{\"$param\"}`s are filled by `bind`, "
          "and by the run's own params.", fields=(
              P("nodes", "list[json]", "The body's nodes, as a protocol graph writes them."),
              P("edges", "list[json]", "The body's edges.", []),
          )),
        P("bind", "map[string, string]",
          "Param name in the body → the record field that fills it, per record.",
          None),
        P("over", "json",
          "The values to map over, in place of the `records` port: a list, "
          "or `{\"range\": [0, 42]}`. Each becomes a one-field record, so a "
          "sweep over layers needs no corpus of integers stored first.",
          None),
        P("as", "string",
          "What `over`'s value is called inside the body — the `$param` it "
          "fills, and the coordinate it lands on.",
          "value"),
        P("collect", "string",
          "`\"stream\"` (flatten every invocation's items), `\"first\"` "
          "(one item per record) or `\"all\"` (nest each invocation's items "
          "under its record).",
          "stream", choices=("stream", "first", "all")),
        P("output", "string",
          "Which of the body's terminal nodes to collect, when it has more "
          "than one.",
          None),
        P("budget_usd", "float",
          "The most the body may spend on provider calls, in US dollars, summed over every invocation. "
          "Required when the body calls a hosted endpoint; the map stops at the record whose call would "
          "pass it. Each body node's own `budget_usd` still bounds that node, and a job-level cap, if one "
          "is set, bounds this one further.",
          None),
    ),
    example={"bind": {"topic": "user"}, "collect": "stream",
             "body": {"nodes": [{"id": "write", "block": "text/generate",
                                 "params": {"model": {"$param": "model"}, "n": 3,
                                            "messages": [{"role": "user",
                                                          "content": {"$param": "topic"}}]}}],
                      "edges": []}},
    example_inputs={"records": {"$ref": {"bench": "you/lab/topics"}}},
)


def read_resume_level(params, inputs=None):
    return read_body_level(params)


def run(ctx, inputs, params):
    records = items_of(inputs.get("records") or [])
    body = params.get("body")
    if not isinstance(body, Mapping) or not body.get("nodes"):
        raise ValueError(
            "records/map needs a `body`: a graph, with `nodes` and "
            "`edges`, written inline and run once per record.")
    bind = dict(params.get("bind") or {})
    over = params.get("over")
    if over is not None:
        if records:
            raise ValueError(
                "records/map takes `over` or a `records` port, not both: "
                "one stream per map")
        name = str(params.get("as") or "value")
        if isinstance(over, Mapping):
            span = over.get("range")
            if not (isinstance(span, (list, tuple)) and len(span) == 2):
                raise ValueError('`over` is a list of values or `{"range": [from, to]}`')
            values: list[Any] = list(range(int(span[0]), int(span[1])))
        elif isinstance(over, (list, tuple)):
            values = list(over)
        else:
            raise ValueError('`over` is a list of values or `{"range": [from, to]}`')
        if not values:
            raise ValueError("`over` names no values: nothing to map")
        records = [{"id": str(v), "coords": {name: v}, name: v} for v in values]
        bind = {name: name, **bind}
    collect = str(params.get("collect", "stream"))
    if collect not in ("stream", "first", "all"):
        raise ValueError(
            f"collect is 'stream', 'first' or 'all', not {collect!r}")
    want = params.get("output")
    cap = params.get("budget_usd")
    pool = ctx.open_budget(cap) if cap is not None else None
    if ctx.on_start:
        ctx.on_start(len(records))

    items: list[dict[str, Any]] = []
    out_arch: dict[str, Any] | None = None
    out_kind: str | None = None
    wired = inputs.get("records")
    record_kind = (item_kind_of(wired) if isinstance(wired, Mapping) else None) or "records/record"
    for rec in records:
        key = str(rec.get("id"))
        if ctx.resume_items and key in ctx.resume_items:
            items.append(ctx.resume_items[key])
            if ctx.on_item:
                ctx.on_item(key, ctx.resume_items[key], True)
            continue
        bound = {name: rec.get(field) for name, field in bind.items()}
        missing_fields = [f for h, f in bind.items() if rec.get(f) is None]
        if missing_fields:
            raise ValueError(
                f"record {key!r} has no {', '.join(missing_fields)} to "
                f"bind into the body's params")
        child_bound = {**(ctx.run_params or {}), **bound}
        try:
            outputs = ctx.sub(body, {"record": collection(record_kind, [rec])}, child_bound, budget=pool)
        except Exception as err:
            if pool is not None and getattr(err, "budget", None) is pool:
                raise err.retold(describe_spend_cap(err, op="records/map", node=ctx.node,
                                                at=f"record {key!r}")) from None
            raise
        if want:
            chosen = outputs.get(str(want))
            if chosen is None:
                raise ValueError(
                    f"the body has no output {want!r}; it ends at "
                    f"{', '.join(sorted(outputs)) or 'nothing'}")
        elif len(outputs) == 1:
            chosen = next(iter(outputs.values()))
        else:
            raise ValueError(
                f"the body ends at {len(outputs)} nodes "
                f"({', '.join(sorted(outputs))}); name one with `output`")
        if collect != "all":
            out_kind = out_kind or item_kind_of(chosen)
        if out_arch is None and isinstance(chosen, Mapping) and isinstance(chosen.get("arch"), Mapping):
            out_arch = dict(chosen["arch"])
        if collect == "stream":
            for sub in items_of(chosen):
                item = dict(sub)
                item["id"] = f"{key}:{sub.get('id')}"
                item["coords"] = {**(rec.get("coords") or {}),
                                  **(sub.get("coords") or {}), "mapped": key}
                items.append(item)
        else:
            rows = items_of(chosen)
            item = {"id": key, "coords": dict(rec.get("coords") or {}),
                    **(dict(rows[0]) if (collect == "first" and rows)
                       else {"items": [dict(r) for r in rows]})}
            item["id"] = key
            items.append(item)
        if ctx.on_item:
            ctx.on_item(key, items[-1], False)
    return collection(
        out_kind or "records/record", items,
        mapped={"records": len(records), "collect": collect,
                "body_nodes": [n.get("id") for n in body.get("nodes", [])]},
        name=params.get("name"), description=params.get("description"),
        **({"arch": out_arch} if out_arch else {}),
        **({"spent_usd": round(pool.spent_usd, 6), "budget_usd": float(cap)} if pool is not None else {}))
