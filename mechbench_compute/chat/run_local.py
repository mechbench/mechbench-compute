from __future__ import annotations

from typing import Any

from mechbench_compute.chat.build_request import build_request
from mechbench_compute.chat.count_by_cause import count_by_cause
from mechbench_compute.chat.count_endings import count_endings
from mechbench_compute.chat.constants import ITEM_KIND, LOCAL
from mechbench_compute.chat.build_item import build_item
from mechbench_compute.chat.describe_reasoning_only import describe_reasoning_only
from mechbench_compute.chat.open_toolbox import open_toolbox
from mechbench_compute.chat.read_local_ending import read_local_ending
from mechbench_compute.chat.read_records import read_records
from mechbench_compute.chat.refuse_remote_only import refuse_remote_only
from mechbench_compute.chat.render_conversation import render_conversation
from mechbench_compute.chat.resolve_sandbox_tools import resolve_sandbox_tools
from mechbench_compute.chat.split_reasoning import find_delimiters, split_reasoning
from mechbench_compute.providers import messages as pm


def run_local(model, ref, records, params, *, inputs=None, on_item=None,
              on_start=None, resume_items=None) -> dict[str, Any]:
    import numpy as _np

    from mechbench_compute import dialects
    from mechbench_compute import intervene as intervene_mod
    from mechbench_compute import tools as tool_mod
    from mechbench_compute.distill import encode, prefill_decision
    from mechbench_compute.generate import sample_completion_cached
    from mechbench_compute.seeds import item_seed

    max_tool_rounds = int(params.get("max_tool_rounds", 3))
    image, tool_specs = resolve_sandbox_tools(params)
    block_runner = params.get("_block_runner")
    tok = model.tokenizer
    dialect = (dialects.dialect_for(tok, model=str(getattr(ref, "base", ref)))
               if tool_specs else None)
    delimiters = find_delimiters(tok)
    model_name = str(getattr(ref, "base", ref))
    refuse_remote_only(params)
    recs = read_records(records)
    n = int(params.get("n", 1))
    start = int(params.get("start", 0))
    seed = params.get("seed", 0)
    temperature = float(params.get("temperature") or 0.9)
    top_p = float(params.get("top_p") or 0.95)
    max_tokens = int(params.get("max_tokens", 1024))
    stop_strings = tuple(params.get("stop") or ())
    model_wire = ref.to_wire() if hasattr(ref, "to_wire") else ref
    plan = intervene_mod.plan(model, params, inputs)
    cells: list = plan.cells if plan else [None]
    if on_start:
        on_start(len(recs) * n * len(cells))
    items: list[dict[str, Any]] = []
    remote_only = [k for k in ("effort", "reasoning_display", "prompt_cache")
                   if params.get(k) is not None]
    if remote_only:
        raise ValueError(
            f"{', '.join(remote_only)} apply to a model a provider runs, not to "
            "local weights; drop them")
    on_tool_error = str(params.get("on_tool_error", "record"))
    if on_tool_error not in ("record", "fail"):
        raise ValueError(
            f"on_tool_error must be 'record' or 'fail', not {on_tool_error!r}")
    tool_errors: list[dict[str, Any]] = []
    responses_with_calls = 0
    for cell, rec in ((c, r) for c in cells for r in recs):
        with intervene_mod.edit_weights(model, plan.weight_items if plan else (),
                                        cell.factor if plan else 0.0):
            req = build_request(rec, params, model=str(getattr(ref, "base", ref)),
                                provider_options={})
            box0 = tool_mod.build_toolbox(tool_specs, block_runner=block_runner)
            hf_tools = [dialects.tool_to_hf(t) for t in box0.tools] if box0 else []
            for k in range(start, start + n):
                key = f"{rec.get('id')}:{k}" + (f":{cell.slug}" if plan else "")
                if resume_items and key in resume_items:
                    items.append(resume_items[key])
                    if on_item:
                        on_item(key, resume_items[key], True)
                    continue
                rng = _np.random.default_rng(item_seed(seed, rec.get("id"), k))
                box, session = open_toolbox(image, tool_specs, block_runner=block_runner)
                turn = req
                called_a_tool = False
                thoughts: list[pm.ReasoningPart] = []
                for round_no in range(max_tool_rounds + 1):
                    ids = encode(tok, render_conversation(
                        tok, turn, tools=hf_tools, dialect=dialect))
                    if plan:
                        prompt_tokens = [tok.decode([int(t)]) for t in ids]
                        prefill = prefill_decision(
                            model, ids, interventions=plan.live(cell, prompt_tokens, rec))
                        live = plan.live(cell, prompt_tokens, rec)
                    else:
                        prefill, live = prefill_decision(model, ids), None
                    text, out_ids = sample_completion_cached(
                        model, ids, max_tokens=max_tokens, temperature=temperature,
                        top_p=top_p, rng=rng, prefill=prefill, return_ids=True,
                        stop_strings=stop_strings,
                        **({"interventions": live} if plan else {}))
                    ended = read_local_ending(tok, out_ids, stop_strings=stop_strings,
                                              max_tokens=max_tokens)
                    thought, text = split_reasoning(text, delimiters)
                    said = tuple(pm.ReasoningPart(text=t, provider=LOCAL, model=model_name)
                                 for t in thought)
                    thoughts.extend(said)
                    if not box or round_no == max_tool_rounds:
                        break
                    text, tool_calls = (dialect.parse(text, box.tools)
                                        if dialect else (text, []))
                    if not tool_calls:
                        break
                    called_a_tool = True
                    results = [box.call(c) for c in tool_calls]
                    turn = turn.with_messages([
                        *turn.messages,
                        pm.Message(role="assistant",
                                   content=(*said, pm.TextPart(text), *tool_calls)),
                        pm.Message(role="tool", content=tuple(results)),
                    ])
                item_errors: list[dict[str, Any]] = []
                if box:
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
                item = build_item(rec, k, text, model_wire=model_wire, params=params,
                                  parts=(*thoughts, pm.TextPart(text)) if thoughts else (),
                                  tool_errors=item_errors,
                                  sampling={"temperature": temperature, "top_p": top_p,
                                            "seed": seed, "index": k,
                                            "ended": "empty" if thoughts and not text else ended},
                                  tool_runs=[r.to_wire() for r in box.runs],
                                  sandbox_calls=(session.calls if session else ()),
                                  sandbox_snapshot=(session.final_wire() if session else None),
                                  cell=cell if plan else None)
                if thoughts and not text:
                    item["metadata"]["empty"] = describe_reasoning_only(model_name, max_tokens)
                items.append(item)
                if on_item:
                    on_item(key, item)
    from mechbench_compute.lexicon import kinds as K

    return K.collection(
        ITEM_KIND, items,
        name=params.get("name", "chat"),
        description=params.get("description", ""),
        fidelity="text",
        ended=count_endings(items),
        **(plan.header() if plan else {}),
        **({"tools": {
            "dialect": dialect.name if dialect else None,
            "responses": len(items),
            "with_calls": responses_with_calls,
            "without_calls": len(items) - responses_with_calls,
            "errors": tool_errors,
            "errors_by_cause": count_by_cause(tool_errors),
        }} if tool_specs else {}),
    )
