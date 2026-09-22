from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.chat.constants import LOCAL
from mechbench_compute.providers import messages as pm


def render_conversation(tokenizer, req: pm.ChatRequest, *,
                        tools: Sequence[Mapping[str, Any]] = (),
                        dialect=None) -> str:
    """A canonical conversation through a local chat template.

    The system prompt merges into the FIRST user turn: several of these
    instruction-tuned templates accept no system role at all.

    Tool parts go through the template rather than into prose: a call as
    a real `tool_calls` entry, a result as a real tool turn, so the
    model reads both in the format it was trained on — and `tools` is
    declared the same way, by the template.

    Reasoning is never prose here either. A turn's reasoning reaches the
    template as `reasoning_content`, the field the templates that mark
    reasoning read, and only when this same model wrote it; the template
    decides whether an earlier turn's reasoning is shown (Gemma 4 shows
    it only on a turn that called a tool). Earlier turns are rendered as
    they were, so the prompt only ever grows.
    """
    from mechbench_compute import dialects as _dl

    turns: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for i, m in enumerate(req.messages):
        calls = [p for p in m.content if isinstance(p, pm.ToolCallPart)]
        results = [p for p in m.content if isinstance(p, pm.ToolResultPart)]
        if results:
            # One turn per result, under the role this model expects:
            # Llama reads them as `ipython`, Qwen and Gemma as `tool`.
            # The wrong role means the model reads its own tool output
            # as though a user had said it.
            for r in results:
                turns.append(_dl.result_message(
                    dialect, names.get(getattr(r, "tool_call_id", ""), ""),
                    str(getattr(r, "content", ""))))
            continue
        text = m.text()
        if i == 0 and m.role == "user" and req.system:
            text = f"{req.system}\n\n{text}"
        turn: dict[str, Any] = {"role": m.role, "content": text}
        thought = [p.text for p in m.content
                   if isinstance(p, pm.ReasoningPart) and p.text
                   and pm.is_replayable(p.provider, p.model, provider=LOCAL,
                                        model=req.model)]
        if thought and m.role == "assistant":
            turn["reasoning_content"] = "\n\n".join(thought)
        if calls:
            for c in calls:
                names[c.id] = c.name
            turn["tool_calls"] = [_dl.call_to_hf(c) for c in calls]
        turns.append(turn)
    if not turns:
        turns = [{"role": "user", "content": req.system}]
    kw: dict[str, Any] = {"tools": list(tools)} if tools else {}
    return tokenizer.apply_chat_template(turns, tokenize=False,
                                         add_generation_prompt=True, **kw)
