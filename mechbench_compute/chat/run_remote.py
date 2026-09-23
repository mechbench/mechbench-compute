from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mechbench_compute.chat.build_item import build_item
from mechbench_compute.chat.build_request import build_request
from mechbench_compute.chat.constants import ITEM_KIND, ON_EMPTY
from mechbench_compute.chat.count_by_cause import count_by_cause
from mechbench_compute.chat.open_toolbox import open_toolbox
from mechbench_compute.chat.read_empty import read_empty
from mechbench_compute.chat.read_records import read_records
from mechbench_compute.chat.resolve_sandbox_tools import resolve_sandbox_tools
from mechbench_compute.chat.summarize_spend import summarize_spend
from mechbench_compute.providers import Budget, build_budget, make_transport
from mechbench_compute.providers import limiter as pl
from mechbench_compute.providers import messages as pm
from mechbench_compute.providers.errors import ProviderError
from mechbench_compute.tools import build_toolbox


def run_remote(ref, records, params, *, secrets=None, cassette=None,
               cassette_mode=None, limiter=None, job_budget: Budget | None = None,
               on_item=None, on_start=None,
               resume_items=None) -> dict[str, Any]:
    """The endpoint path. `ref` is an endpoint ModelRef.

    An empty reply is checkpointed through `on_item` like any other, so
    a resumed run never buys it twice; `on_empty` is applied when the
    output collection is assembled, to resumed items as well."""
    from mechbench_compute.seeds import item_seed

    on_empty = str(params.get("on_empty", "keep"))
    if on_empty not in ON_EMPTY:
        raise ValueError(f"on_empty is one of {ON_EMPTY}, not {on_empty!r}")

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
        cassette_mode=str(cassette_mode
                          or params.get("cassette_mode", "replay")))
    budget = build_budget(params)
    if job_budget is not None:
        # The node's cap under the job's: a graph whose node caps sum
        # past the job's cap still cannot spend past the job's.
        budget = job_budget.child(budget.cap_usd)
    # Which of the provider's APIs answers: the node's choice, else the
    # model's default (the Responses API only where a model needs it).
    api = params.get("api")
    if api is None:
        from mechbench_compute.providers.openai_responses import default_api

        api = default_api(provider, ref.base)
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

    # Tools are ordinary blocks; each item gets its own toolbox so the
    # runs it records are its own even under the pool. A `sandbox` image
    # adds its tools and a per-item session.
    image, tool_specs = resolve_sandbox_tools(params)
    max_tool_rounds = int(params.get("max_tool_rounds", 3))
    block_runner = params.get("_block_runner")

    specs = build_toolbox(tool_specs).specs()

    recs = read_records(records)
    if on_start:
        on_start(len(recs) * n)

    plan: list[tuple[str, dict, int, pm.ChatRequest]] = []
    # Items land in completion order; the executor stores a collection
    # in key order, so the order here is nobody's contract.
    items: list[Any] = []
    replayed = 0
    calls: list[dict[str, Any]] = []
    model_wire = ref.to_wire()

    for i, rec in enumerate(recs):
        for k in range(start, start + n):
            key = f"{rec.get('id')}:{k}"
            if resume_items and key in resume_items:
                # Exchangeable: this item was paid for once, by this
                # same process identity. Reuse it rather than buy it
                # again.
                items.append(resume_items[key])
                if on_item:
                    on_item(key, resume_items[key], True)
                if on_empty == "error" and read_empty(resume_items[key]):
                    raise ProviderError(read_empty(resume_items[key])["message"])
                continue
            item_sd = (item_seed(seed, rec.get("id"), k)
                       if seed is not None and transport.capabilities.seed
                       and api != "responses"   # it has no seed field
                       else None)
            plan.append((key, dict(rec), k,
                         build_request(rec, params, model=ref.base,
                                       provider_options=options,
                                       seed=item_sd, tools=specs, api=api)))

    def one(entry):
        key, rec, k, req = entry
        box, session = open_toolbox(image, tool_specs, block_runner=block_runner)
        extra_calls: list[Any] = []
        rounds: list[pm.Message] = []
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
            rounds.append(out.as_message())
            req = req.with_messages([
                *req.messages, out.as_message(),
                pm.Message(role="user", content=tuple(results)),
            ])
        item = build_item(rec, k, out.text, model_wire=model_wire, params=params,
                          parts=out.parts, call=out.call.to_wire(),
                          sampling={"temperature": params.get("temperature"),
                                    "max_tokens": int(params.get("max_tokens", 1024)),
                                    "seed": req.seed, "index": k},
                          tool_runs=[r.to_wire() for r in box.runs],
                          rounds=rounds,
                          sandbox_calls=(session.calls if session else ()),
                          sandbox_snapshot=(session.final_wire() if session else None))
        if out.empty is not None:
            item["metadata"]["empty"] = out.empty.to_wire()
        return key, item, out.call, extra_calls

    failed: list[ProviderError] = []
    if plan:
        def land(result) -> None:
            key, item, call, extra = result
            items.append(item)
            for c in (call, *extra):
                calls.append(c.to_wire())
            nonlocal replayed
            replayed += int(bool(call.replayed))
            if on_item:
                on_item(key, item)
            if on_empty == "error" and read_empty(item) and not failed:
                failed.append(ProviderError(read_empty(item)["message"]))

        if concurrency == 1 or len(plan) == 1:
            for entry in plan:
                land(one(entry))
                if failed:
                    break
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(one, e) for e in plan]
                from concurrent.futures import as_completed

                for fut in as_completed(futures):
                    if fut.cancelled():
                        continue
                    land(fut.result())
                    if failed:
                        # Calls not yet started are not bought; those in
                        # flight are paid for, so they still land.
                        for f in futures:
                            f.cancel()
    if failed:
        raise failed[0]

    empties = sorted((it for it in items if read_empty(it)),
                     key=lambda it: str(it.get("id")))
    if on_empty == "skip":
        items = [it for it in items if not read_empty(it)]

    from mechbench_compute.lexicon import kinds as K

    return K.collection(
        ITEM_KIND, items,
        name=params.get("name", "chat"),
        description=params.get("description", ""),
        fidelity="text",
        spend=summarize_spend(calls, budget, provider=provider, dry_run=dry_run,
                              replayed=replayed),
        empty={"count": len(empties),
               "by_cause": count_by_cause([read_empty(it) for it in empties]),
               "ids": [str(it.get("id")) for it in empties],
               "policy": on_empty},
    )

