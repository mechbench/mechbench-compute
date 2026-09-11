"""`~canonical/ops/chat/1` — one block for local weights and remote
endpoints (task 000337, epic 000334).

The same node, the same params, the same output kind, whichever side of
the network answers:

    model: "google/gemma-3-4b-it"                → MLX, on this machine
    model: {provider: "anthropic", model: ...}   → the Messages API

That is the point of the endpoint ModelRef. A protocol that compares a
local fine-tune against a frontier model is one graph with two chat
nodes, not two systems, and everything downstream — text/stats,
group-stats, eval — reads the same document collection either way.

What differs is what the run can promise. Local sampling is
`reproducible`: seeds make it a pure function of its item key. A remote
call is `exchangeable` at best — someone else's sampler, someone else's
weights, possibly a different dated model version tomorrow — so the
manifest records the version that ANSWERED, the cost, and the usage,
and resume tops the set up rather than claiming bit-identity.

Remote items are I/O bound, so they run on a bounded thread pool.
Output order stays canonical (records × sample index) regardless of
completion order; the SPOOL receives items as they land, which is what
lets an interrupted node keep what it paid for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from mechbench_compute.providers import Budget, budget_from, make_transport
from mechbench_compute.providers import limiter as pl
from mechbench_compute.providers import messages as pm
from mechbench_compute.tools import Toolbox, toolbox_from

#: What a chat node emits per item, for both paths.
ITEM_KIND = "~canonical/kinds/text"


def _records(value: Any) -> list[dict[str, Any]]:
    # `blocks._records` understands document collections since the
    # vector path needed it too; this stays as the module's name for it.
    from mechbench_compute.blocks import _records as base_records

    return base_records(value)


def build_request(rec: Mapping[str, Any], params: Mapping[str, Any], *,
                  model: str, provider_options: Mapping[str, Any],
                  seed: int | None = None,
                  tools: Sequence[Any] | None = None) -> pm.ChatRequest:
    """One record's request. A record carries either a full `messages`
    conversation or the `system`/`user` fields a corpus record has —
    the same fields `generate` reads, so a protocol can swap a local
    generate node for a chat node without rewriting its corpus."""
    f_system = params.get("system_field", "system")
    f_user = params.get("user_field", "user")
    f_messages = params.get("messages_field", "messages")
    convo = rec.get(f_messages)
    if convo is None and f_user in rec:
        convo = [{"role": "user", "content": rec[f_user]}]
    if convo is None:
        convo = params.get("messages") or []
    system = str(rec.get(f_system) or params.get("system") or "")
    return pm.request({
        "model": model,
        "system": system,
        "messages": convo,
        "max_tokens": int(params.get("max_tokens", 1024)),
        "tools": tuple(tools) if tools is not None else (),
        "tool_choice": params.get("tool_choice"),
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "stop": tuple(params.get("stop") or ()),
        "seed": seed,
        "json_mode": bool(params.get("json_mode", False)),
        "logprobs": params.get("logprobs"),
        "provider_options": dict(provider_options),
    })


def _item(rec: Mapping[str, Any], k: int, text: str, *,
          model_wire: Any, params: Mapping[str, Any],
          parts: Sequence[Any] = (), call: Mapping[str, Any] | None = None,
          sampling: Mapping[str, Any] | None = None,
          tool_runs: Sequence[Any] = ()) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "coords": {**(rec.get("coords") or {}), "sample": k},
        "model": model_wire,
    }
    if sampling:
        meta["sampling"] = dict(sampling)
    if call is not None:
        meta["call"] = dict(call)
    if tool_runs:
        # What the model actually did with the capabilities it was
        # given — the point of offering them.
        meta["tool_runs"] = [dict(r) for r in tool_runs]
    tool_parts = [p.to_wire() for p in parts
                  if not isinstance(p, pm.TextPart)]
    if tool_parts:
        meta["parts"] = tool_parts
    item = {"id": f"{rec.get('id')}-s{k}", "kind": ITEM_KIND, "text": text,
            "metadata": meta}
    if params.get("keep_fields"):
        for f in params["keep_fields"]:
            if f in rec:
                item[f] = rec[f]
    return item


def _summary(calls: Sequence[Mapping[str, Any]], budget: Budget, *,
             provider: str, dry_run: bool, replayed: int) -> dict[str, Any]:
    """The node's spend, which the manifest sums (000324's partials
    carry the running total). A reader should never have to add up
    per-item records to learn what a node cost."""
    usage: dict[str, int] = {}
    versions: dict[str, int] = {}
    for c in calls:
        for k, v in (c.get("usage") or {}).items():
            usage[k] = usage.get(k, 0) + int(v)
        mv = str(c.get("model_version") or "")
        versions[mv] = versions.get(mv, 0) + 1
    return {
        "provider": provider,
        "calls": len(calls),
        "cost_usd": round(sum(float(c.get("cost_usd", 0.0)) for c in calls), 8),
        "unpriced_calls": sum(1 for c in calls if c.get("priced") is False),
        "usage": usage,
        "model_versions": versions,
        "budget": budget.to_wire(),
        "throttled_seconds": round(
            sum(float(c.get("throttled_seconds", 0.0)) for c in calls), 3),
        "dry_run": bool(dry_run),
        "replayed": replayed,
    }


def run_remote(ref, records, params, *, secrets=None, cassette=None,
               limiter=None, job_budget: Budget | None = None,
               on_item=None, on_start=None,
               resume_items=None) -> dict[str, Any]:
    """The endpoint path. `ref` is an endpoint ModelRef."""
    from mechbench_compute.seeds import item_seed

    provider = ref.provider
    creds = dict((secrets or {}).get(provider) or {})
    dry_run = bool(params.get("dry_run", False))
    tape = None
    if cassette is not None:
        from mechbench_compute.providers.cassette import Cassette

        tape = (cassette if hasattr(cassette, "entries")
                else Cassette.from_wire(cassette))
    transport = make_transport(
        provider, creds or None, base_url=params.get("base_url"),
        dry_run=dry_run and tape is None, cassette=tape,
        cassette_mode=str(params.get("cassette_mode", "replay")))
    budget = budget_from(params)
    if job_budget is not None:
        # The node's cap under the job's: a graph whose node caps sum
        # past the job's cap still cannot spend past the job's.
        budget = job_budget.child(budget.cap_usd)
    options = {**dict(ref.provider_options or {}),
               **dict(params.get("provider_options") or {})}
    n = int(params.get("n", 1))
    start = int(params.get("start", 0))
    seed = params.get("seed")
    concurrency = max(1, int(params.get("concurrency", 4)))
    record_requests = bool(params.get("record_requests", False))
    # Rate limits are per ACCOUNT, so the limiter's scope is the
    # credential's fingerprint — two keys for one provider get two
    # buckets, and nothing persisted names a secret.
    scope = str(params.get("limit_scope") or pl.scope_for(provider, creds))

    # Tools are ordinary blocks (task 000340); each item gets its own
    # toolbox so the runs it records are its own even under the pool.
    tool_specs = params.get("tools") or ()
    max_tool_rounds = int(params.get("max_tool_rounds", 3))
    block_runner = params.get("_block_runner")

    def new_toolbox() -> Toolbox:
        return toolbox_from(tool_specs, block_runner=block_runner)

    specs = new_toolbox().specs()

    recs = _records(records)
    if on_start:
        on_start(len(recs) * n)

    plan: list[tuple[int, str, dict, int, pm.ChatRequest]] = []
    items: list[Any] = [None] * (len(recs) * n)
    replayed = 0
    calls: list[dict[str, Any]] = []
    model_wire = ref.to_wire()

    for i, rec in enumerate(recs):
        for j, k in enumerate(range(start, start + n)):
            pos = i * n + j
            key = f"{rec.get('id')}:{k}"
            if resume_items and key in resume_items:
                # Exchangeable (epic 000320): this item was paid for
                # once, by this same process identity. Reuse it rather
                # than buy it again.
                items[pos] = resume_items[key]
                if on_item:
                    on_item(key, resume_items[key], True)
                continue
            item_sd = (item_seed(seed, rec.get("id"), k)
                       if seed is not None and transport.capabilities.seed
                       else None)
            plan.append((pos, key, dict(rec), k,
                         build_request(rec, params, model=ref.base,
                                       provider_options=options,
                                       seed=item_sd, tools=specs)))

    def one(entry):
        pos, key, rec, k, req = entry
        box = new_toolbox()
        extra_calls: list[Any] = []
        # The tool loop: answer, run what it asked for, hand back the
        # results, ask again — bounded, because a model and its tools
        # can talk to each other for a long time at someone's expense.
        for round_no in range(max_tool_rounds + 1):
            out = transport.chat(req, budget=budget, limiter=limiter, scope=scope,
                                 record_request=record_requests)
            if round_no:
                extra_calls.append(out.call)
            if not (out.tool_calls and box) or round_no == max_tool_rounds:
                break
            results = [box.call(c) for c in out.tool_calls]
            req = req.with_messages([
                *req.messages, out.as_message(),
                pm.Message(role="user", content=tuple(results)),
            ])
        item = _item(rec, k, out.text, model_wire=model_wire, params=params,
                     parts=out.parts, call=out.call.to_wire(),
                     sampling={"temperature": params.get("temperature"),
                               "max_tokens": int(params.get("max_tokens", 1024)),
                               "seed": req.seed, "index": k},
                     tool_runs=[r.to_wire() for r in box.runs])
        return pos, key, item, out.call, extra_calls

    if plan:
        def land(result) -> None:
            pos, key, item, call, extra = result
            items[pos] = item
            for c in (call, *extra):
                calls.append(c.to_wire())
            nonlocal replayed
            replayed += int(bool(call.replayed))
            if on_item:
                on_item(key, item)

        if concurrency == 1 or len(plan) == 1:
            for entry in plan:
                land(one(entry))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(one, e) for e in plan]
                # Completion order for the spool (an item is safe as
                # soon as it lands), canonical order in the output.
                from concurrent.futures import as_completed

                for fut in as_completed(futures):
                    land(fut.result())

    return {
        "kind": "document_collection",
        "name": params.get("name", "chat"),
        "description": params.get("description", ""),
        "fidelity": "text",
        "item_kind": ITEM_KIND,
        "items": [it for it in items if it is not None],
        "spend": _summary(calls, budget, provider=provider, dry_run=dry_run,
                          replayed=replayed),
    }


def run_local(model, ref, records, params, *, on_item=None, on_start=None,
              resume_items=None) -> dict[str, Any]:
    """The MLX path: the same block, the same output, sampled here.

    Multi-turn conversations render through the tokenizer's own chat
    template, so a local model sees the same turns the remote one did —
    the comparison a protocol is usually built to make.
    """
    import numpy as _np

    from mechbench_compute import tools as tool_mod
    from mechbench_compute.distill import encode, prefill_decision
    from mechbench_compute.generate import sample_completion_cached
    from mechbench_compute.seeds import item_seed

    # Local weights have no tool protocol, so the tools are described
    # in the system prompt and the calls are read back out of the text
    # (task 000340). Same tools, same handlers, same provenance — only
    # the transport differs.
    family = str(params.get("tool_family", "gemma"))
    max_tool_rounds = int(params.get("max_tool_rounds", 3))
    tool_specs = params.get("tools") or ()
    block_runner = params.get("_block_runner")
    tok = model.tokenizer
    recs = _records(records)
    n = int(params.get("n", 1))
    start = int(params.get("start", 0))
    seed = params.get("seed", 0)
    temperature = float(params.get("temperature") or 0.9)
    top_p = float(params.get("top_p") or 0.95)
    max_tokens = int(params.get("max_tokens", 1024))
    model_wire = ref.to_wire() if hasattr(ref, "to_wire") else ref
    if on_start:
        on_start(len(recs) * n)
    items: list[dict[str, Any]] = []
    #: Tool-shaped responses that produced no call (000437).
    near_misses = 0
    for rec in recs:
        req = build_request(rec, params, model=str(getattr(ref, "base", ref)),
                            provider_options={})
        box0 = tool_mod.toolbox_from(tool_specs, block_runner=block_runner)
        if box0:
            # The tools go where a local model can actually see them.
            req = replace(req, system="\n\n".join(
                x for x in (req.system,
                            tool_mod.render_tools(box0.tools, family=family))
                if x))
        for k in range(start, start + n):
            key = f"{rec.get('id')}:{k}"
            if resume_items and key in resume_items:
                items.append(resume_items[key])
                if on_item:
                    on_item(key, resume_items[key], True)
                continue
            rng = _np.random.default_rng(item_seed(seed, rec.get("id"), k))
            box = tool_mod.toolbox_from(tool_specs, block_runner=block_runner)
            turn = req
            called_a_tool = False
            for round_no in range(max_tool_rounds + 1):
                ids = encode(tok, render_conversation(tok, turn))
                text = sample_completion_cached(
                    model, ids, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, rng=rng, prefill=prefill_decision(model, ids))
                if not box or round_no == max_tool_rounds:
                    break
                text, tool_calls = tool_mod.parse_tool_calls(
                    text, tools=box.tools, family=family)
                if not tool_calls:
                    break
                called_a_tool = True
                results = [box.call(c) for c in tool_calls]
                turn = turn.with_messages([
                    *turn.messages,
                    pm.Message(role="assistant",
                               content=(pm.TextPart(text), *tool_calls)),
                    pm.Message(role="user", content=tuple(results)),
                ])
            # A response that was reaching for a tool and produced no
            # call is a NEAR MISS, not a plain answer. Counting them is
            # the whole lesson of 000437: a correct call the parser did
            # not recognize looked exactly like no call at all, across
            # 320 generations, with nothing to notice.
            if box and not called_a_tool and tool_mod.looks_like_a_tool_call(text):
                near_misses += 1
            item = _item(rec, k, text, model_wire=model_wire, params=params,
                         sampling={"temperature": temperature, "top_p": top_p,
                                   "seed": seed, "index": k},
                         tool_runs=[r.to_wire() for r in box.runs])
            items.append(item)
            if on_item:
                on_item(key, item)
    return {
        "kind": "document_collection",
        "name": params.get("name", "chat"),
        "description": params.get("description", ""),
        "fidelity": "text",
        "item_kind": ITEM_KIND,
        "items": items,
        # Reported even when zero: "no tool calls" and "no tool calls
        # and nobody tried" are different facts about a run.
        **({"tool_near_misses": near_misses} if tool_specs else {}),
    }


def render_conversation(tokenizer, req: pm.ChatRequest) -> str:
    """A canonical conversation through a local chat template.

    The system prompt merges into the FIRST user turn, which is how
    `render_chat` has always driven these instruction-tuned templates
    (several of them accept no system role at all); tool parts are
    rendered as plain text, since a local model here has no tool
    protocol to receive them with.
    """
    turns: list[dict[str, str]] = []
    for i, m in enumerate(req.messages):
        text = m.text()
        extra = [p for p in m.content if not isinstance(p, pm.TextPart)]
        if extra:
            text = "\n".join([text, *(str(p.to_wire()) for p in extra)]).strip()
        if i == 0 and m.role == "user" and req.system:
            text = f"{req.system}\n\n{text}"
        turns.append({"role": m.role, "content": text})
    if not turns:
        turns = [{"role": "user", "content": req.system}]
    return tokenizer.apply_chat_template(turns, tokenize=False,
                                         add_generation_prompt=True)
