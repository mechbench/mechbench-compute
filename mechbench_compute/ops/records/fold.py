from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.protocol.protocol_spec import ProtocolSpec

OP = Op(
    name="records/fold",
    summary=(
        "Run a body graph step after step, each step reading the state the "
        "last one wrote — the loop a conversation, a refinement or an "
        "agentic round is made of — until the steps run out or the state "
        "says stop."
    ),
    description="""\
`records/map` runs a body once per record with no memory between runs;
this runs it once per **step**, and what the body produces at step *t* is
what it reads at step *t + 1*. The state arrives on the `state` port,
enters the body on the body input named `state` (an edge `{"from":
{"input": "state"}}`), and comes back out of the body's output named by
`output` (or its one output). The final state is the result.

**Steps.** `over` is a list of objects, one per step, cycled when `steps`
is longer than it: each object's keys are `$param`s the body's nodes read
that step — `[{"participant": "ana"}, {"participant": "bo"}]` with `steps:
6` is a six-turn round robin. A body that needs no per-step values takes
`steps` alone. The step's index is bound as `$step`.

**Stopping.** `until: {"field": "stopped"}` ends the fold early when every
item of the state has a non-empty value in that field — which is how a
body says the conversation reached its stop phrase (`text/extend` writes
`stopped`), the answer converged, or the tool loop finished. The header's
`folded` says how many steps ran and why it ended.

Every step is an item keyed by its index, so an interrupted fold resumes
at the step it reached with the state it had. A body sees one state at a
time and nothing else.

The conversation: `text/render` → `text/chat` → `text/extend` as the
body, transcripts as the state, participants as `over` — a multi-party
conversation is three ops and a loop, not an operation of its own.
""",
    inputs=(
        In("state", "collection",
           "The starting state: what the body reads at step 0 — any "
           "collection, of whatever kind the body's `state` input takes.",
           many=True),
    ),
    output=Output('records/record', collection=True, doc="The state after the last step — the body's `output` at that step, its kind whatever the body's output node emits — with `folded` on the header: `steps` (how many ran), `stopped` (`\"steps\"`, or `\"until\"` when the state said stop), `body_nodes`."),
    params=(
        P("body", "object",
          "The graph to run per step — `{nodes, edges}`, the same shape a "
          "protocol's graph has. An edge from `{\"input\": \"state\"}` "
          "carries the state in; `output` names the node that carries it out.",
          fields=(
              P("nodes", "list[json]", "The body's nodes, as a protocol graph writes them."),
              P("edges", "list[json]", "The body's edges.", []),
          )),
        P("over", "list[json]",
          "One object per step, its keys the `$param`s the body reads that "
          "step — open by design, since they are the body's names; cycled "
          "when `steps` exceeds its length.",
          None),
        P("steps", "int",
          "How many steps to run. Defaults to the length of `over`.",
          None),
        P("until", "object",
          "Stop early when every state item has a non-empty value in "
          "`field`.",
          None, fields=(P("field", "string", "The state field that says stop."),)),
        P("output", "string",
          "Which of the body's terminal nodes carries the state out, when "
          "it has more than one.",
          None),
    ),
    example={"over": [{"participant": "ana"}, {"participant": "bo"}], "steps": 6,
             "until": {"field": "stopped"}, "output": "next",
             "body": {"nodes": [
                 {"id": "view", "block": "text/render", "params": {"participant": {"$param": "participant"}}},
                 {"id": "say", "block": "text/chat", "params": {"model": {"$param": "model"}}},
                 {"id": "next", "block": "text/extend", "params": {"participant": {"$param": "participant"}}}],
                      "edges": [
                 {"from": {"input": "state"}, "to": {"node": "view", "port": "transcripts"}},
                 {"from": {"node": "view"}, "to": {"node": "say", "port": "records"}},
                 {"from": {"input": "state"}, "to": {"node": "next", "port": "transcripts"}},
                 {"from": {"node": "say"}, "to": {"node": "next", "port": "replies"}}]}},
    example_inputs={"state": {"$ref": {"bench": "you/lab/openings"}}},
)


def run(ctx, inputs, params):
    """`records/fold` (task 000617): run a body step after step, each
    step reading the state the last one wrote.

    `records/map` runs its body once per record with no memory
    between runs; a conversation, a refinement, an agentic round
    is the other shape — the body's output at step t is its input
    at step t + 1. The state enters the body on its `state` input
    (an edge from `{"input": "state"}`), and leaves by the body
    output named `output`. `over` binds one object of `$param`s per
    step, cycled; `until.field` stops the fold when every state item
    has that field set.

    Every step is an item keyed by its index and spooled with the
    state it produced, so an interrupted fold resumes at the step it
    reached — the resume machinery treats steps exactly as it treats
    a chat node's items. The body sees one state and nothing else.
    """
    from mechbench_compute import dataflow as dataflow_mod
    from mechbench_compute.lexicon import kinds as K

    state = inputs.get("state")
    if state is None:
        raise ValueError("records/fold needs a starting `state` on its port")
    body = params.get("body")
    if not isinstance(body, Mapping) or not body.get("nodes"):
        raise ValueError(
            "records/fold needs a `body`: a graph, with `nodes` and "
            "`edges`, run once per step")
    # A fold's body is a declared graph by construction: the state
    # reaches it by an edge from `{"input": "state"}`, which only the
    # declared form has.
    body = {**body, "dataflow": dataflow_mod.DATAFLOW}
    over = params.get("over")
    if over is not None and not (isinstance(over, list)
                                 and all(isinstance(o, Mapping) for o in over)):
        raise ValueError("`over` is a list of objects, one per step")
    over = [dict(o) for o in (over or [])]
    steps = params.get("steps")
    steps = len(over) if steps is None else int(steps)
    if steps < 1:
        raise ValueError("a fold runs at least one step: give `over` or `steps`")
    if steps > len(over) and not over:
        over = [{}]
    until = params.get("until") or {}
    stop_field = str(until["field"]) if isinstance(until, Mapping) and until.get("field") else None
    want = params.get("output")
    if ctx.on_start:
        ctx.on_start(steps)

    child = type(ctx.executor)(
        on_download=ctx.executor._on_download,
        on_download_bytes=ctx.executor._on_download_bytes,
        limiter=ctx.executor._limiter, budget=ctx.executor._budget)
    child._model, child._model_id = ctx.executor._model, ctx.executor._model_id

    def says_stop(value: Any) -> bool:
        items = K.items_of(value) if K.item_kind_of(value) else []
        return bool(items) and all(bool(it.get(stop_field)) for it in items)

    ran, stopped = 0, "steps"
    for t in range(steps):
        key = f"step:{t}"
        if ctx.resume_items and key in ctx.resume_items:
            state = ctx.resume_items[key]
            ran += 1
            if ctx.on_item:
                ctx.on_item(key, state, True)
            if stop_field and says_stop(state):
                stopped = "until"
                break
            continue
        step_params = {**(ctx.bindings or {}), **over[t % len(over)], "step": t}
        out = child.run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": body, "bindings": step_params,
                   "params": step_params, "inputs": {"state": state}}),
            secrets=ctx.secrets)
        outputs = out.payload.get("outputs") or {}
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
        state = chosen
        ran += 1
        if ctx.on_item:
            ctx.on_item(key, state, False)
        if stop_field and says_stop(state):
            stopped = "until"
            break
    ctx.executor._model, ctx.executor._model_id = child._model, child._model_id
    if not isinstance(state, Mapping):
        raise ValueError("the body's output is not a collection; a fold's state is one")
    return {**dict(state),
            "folded": {"steps": ran, "stopped": stopped,
                       "body_nodes": [n.get("id") for n in body.get("nodes", [])]},
            **({"name": params["name"]} if params.get("name") else {}),
            **({"description": params["description"]} if params.get("description") else {})}
