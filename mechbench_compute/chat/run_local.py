from __future__ import annotations

from typing import Any

from mechbench_compute.chat.build_request import build_request
from mechbench_compute.chat.by_cause import _by_cause
from mechbench_compute.chat.constants import ITEM_KIND
from mechbench_compute.chat.item import _item
from mechbench_compute.chat.new_box import _new_box
from mechbench_compute.chat.records import _records
from mechbench_compute.chat.refuse_remote_only import _refuse_remote_only
from mechbench_compute.chat.render_conversation import render_conversation
from mechbench_compute.chat.sandbox_and_tools import _sandbox_and_tools
from mechbench_compute.providers import messages as pm


def run_local(model, ref, records, params, *, inputs=None, on_item=None,
              on_start=None, resume_items=None) -> dict[str, Any]:
    """The MLX path: the same block, the same output, sampled here.

    Multi-turn conversations render through the tokenizer's own chat
    template, so a local model sees the same turns the remote one did —
    the comparison a protocol is usually built to make.
    """
    import numpy as _np

    from mechbench_compute import dialects
    from mechbench_compute import intervene as intervene_mod
    from mechbench_compute import tools as tool_mod
    from mechbench_compute.distill import encode, prefill_decision
    from mechbench_compute.generate import sample_completion_cached
    from mechbench_compute.seeds import item_seed

    # The model's own chat template carries the tool protocol (epic
    # 000439): it declares the tools, renders the call, and renders the
    # result. Same tools, same handlers, same provenance as the remote
    # path — only the transport differs.
    max_tool_rounds = int(params.get("max_tool_rounds", 3))
    image, tool_specs = _sandbox_and_tools(params)
    block_runner = params.get("_block_runner")
    tok = model.tokenizer
    # The model's own chat template decides how tools are declared,
    # called and answered (epic 000439). If it has no protocol, this
    # raises rather than inventing one — a model that cannot receive a
    # declaration produces output indistinguishable from a model that
    # chose not to call anything, which is how 024 lost an arm.
    dialect = (dialects.dialect_for(tok, model=str(getattr(ref, "base", ref)))
               if tool_specs else None)
    _refuse_remote_only(params)
    recs = _records(records)
    n = int(params.get("n", 1))
    start = int(params.get("start", 0))
    seed = params.get("seed", 0)
    temperature = float(params.get("temperature") or 0.9)
    top_p = float(params.get("top_p") or 0.95)
    max_tokens = int(params.get("max_tokens", 1024))
    stop_strings = tuple(params.get("stop") or ())
    model_wire = ref.to_wire() if hasattr(ref, "to_wire") else ref
    # An intervention (000601) makes the node a sweep: one set of replies
    # per cell, its axes coordinates, weight edits scoped per strength.
    # Without one the loop is the one it always was.
    plan = intervene_mod.plan(model, params, inputs)
    cells: list = plan.cells if plan else [None]
    if on_start:
        on_start(len(recs) * n * len(cells))
    items: list[dict[str, Any]] = []
    # Tool-call errors, reported and not merely counted. An individual
    # failure does not fail the run unless asked to: `on_tool_error`
    # mirrors group-stats' `on_missing` rather than inventing a second
    # idiom for the same choice.
    on_tool_error = str(params.get("on_tool_error", "record"))
    if on_tool_error not in ("record", "fail"):
        raise ValueError(
            f"on_tool_error must be 'record' or 'fail', not {on_tool_error!r}")
    tool_errors: list[dict[str, Any]] = []
    responses_with_calls = 0
    for cell, rec in ((c, r) for c in cells for r in recs):
        with intervene_mod.edited(model, plan.weight_items if plan else (),
                                  cell.factor if plan else 0.0):
            req = build_request(rec, params, model=str(getattr(ref, "base", ref)),
                                provider_options={})
            box0 = tool_mod.toolbox_from(tool_specs, block_runner=block_runner)
            hf_tools = [dialects.tool_to_hf(t) for t in box0.tools] if box0 else []
            for k in range(start, start + n):
                key = f"{rec.get('id')}:{k}" + (f":{cell.slug}" if plan else "")
                if resume_items and key in resume_items:
                    items.append(resume_items[key])
                    if on_item:
                        on_item(key, resume_items[key], True)
                    continue
                rng = _np.random.default_rng(item_seed(seed, rec.get("id"), k))
                box, session = _new_box(image, tool_specs, block_runner=block_runner)
                turn = req
                called_a_tool = False
                for round_no in range(max_tool_rounds + 1):
                    ids = encode(tok, render_conversation(
                        tok, turn, tools=hf_tools, dialect=dialect))
                    # Every round is its own sequence — a tool result
                    # lengthens the prompt — so each gets a live
                    # intervention over its own tokens (000601).
                    if plan:
                        prompt_tokens = [tok.decode([int(t)]) for t in ids]
                        prefill = prefill_decision(
                            model, ids, interventions=plan.live(cell, prompt_tokens, rec))
                        live = plan.live(cell, prompt_tokens, rec)
                    else:
                        prefill, live = prefill_decision(model, ids), None
                    text = sample_completion_cached(
                        model, ids, max_tokens=max_tokens, temperature=temperature,
                        top_p=top_p, rng=rng, prefill=prefill,
                        stop_strings=stop_strings,
                        **({"interventions": live} if plan else {}))
                    if not box or round_no == max_tool_rounds:
                        break
                    # The call markup leaves the text: it goes back into
                    # the transcript as a structured `tool_calls` entry,
                    # and a model handed its own call twice answers with
                    # nothing.
                    # Only calls to tools we offered are stripped and
                    # executed; an unknown name stays in the text so the
                    # error can quote it.
                    text, tool_calls = (dialect.parse(text, box.tools)
                                        if dialect else (text, []))
                    if not tool_calls:
                        break
                    called_a_tool = True
                    results = [box.call(c) for c in tool_calls]
                    turn = turn.with_messages([
                        *turn.messages,
                        pm.Message(role="assistant",
                                   content=(pm.TextPart(text), *tool_calls)),
                        pm.Message(role="tool", content=tuple(results)),
                    ])
                # A response that was reaching for a tool and produced no
                # call is a NEAR MISS, not a plain answer. Counting them is
                # the whole lesson of 000437: a correct call the parser did
                # not recognize looked exactly like no call at all, across
                # 320 generations, with nothing to notice.
                item_errors: list[dict[str, Any]] = []
                if box:
                    # A tool that ran and raised is an error too, and was
                    # previously only visible per-item in `tool_runs`.
                    for r in box.runs:
                        if r.error:
                            item_errors.append(dialects.ToolError(
                                "execution_failed", r.error, r.tool).to_wire())
                    if called_a_tool:
                        responses_with_calls += 1
                    else:
                        failed = dialects.call_error(text, box.tools, dialect)
                        if failed is not None:
                            item_errors.append(failed.to_wire())
                if item_errors:
                    if on_tool_error == "fail":
                        raise RuntimeError(
                            f"tool call failed on {rec.get('id')!r}: "
                            f"{item_errors[0]['cause']} — {item_errors[0]['detail']}")
                    tool_errors.extend({**e, "item": key} for e in item_errors)
                item = _item(rec, k, text, model_wire=model_wire, params=params,
                             tool_errors=item_errors,
                             sampling={"temperature": temperature, "top_p": top_p,
                                       "seed": seed, "index": k},
                             tool_runs=[r.to_wire() for r in box.runs],
                             sandbox_calls=(session.calls if session else ()),
                             sandbox_snapshot=(session.final_wire() if session else None),
                             cell=cell if plan else None)
                items.append(item)
                if on_item:
                    on_item(key, item)
    from mechbench_compute.lexicon import kinds as K

    return K.collection(
        ITEM_KIND, items,
        name=params.get("name", "chat"),
        description=params.get("description", ""),
        fidelity="text",
        **(plan.header() if plan else {}),
        # Reported even when zero: "no tool calls" and "no tool calls
        # and nobody tried" are different facts about a run.
        # Everything a reader needs to know about tool use, without
        # re-deriving it from the items: how many responses called a
        # tool at all (a statistic — answering directly is a legitimate
        # outcome, not an error), and every error with its cause.
        **({"tools": {
            "dialect": dialect.name if dialect else None,
            "responses": len(items),
            "with_calls": responses_with_calls,
            "without_calls": len(items) - responses_with_calls,
            "errors": tool_errors,
            "errors_by_cause": _by_cause(tool_errors),
        }} if tool_specs else {}),
    )
