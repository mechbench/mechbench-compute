"""The Responses API shape, for OpenAI and xAI.

Chat Completions returns an OpenAI reasoning model's reasoning as a
token count and an xAI model's as text it has no field to take back.
The Responses API returns reasoning as output ITEMS — `{"type":
"reasoning", "id", "summary", "encrypted_content", …}` — which the
provider's documentation says to hand back unchanged in the next
request's `input`, with every other output item, so the model keeps its
reasoning across turns and through a tool loop.

This module is that wire mapping; the transport, the credential, the
host table, the prices and the limiter are the provider's own
(`openai_compatible.py`), because it is the same provider answering
through another door. A request reaches it by `api: "responses"`.

Mapping notes that matter:

- Requests are stateless: `store: false`, nothing kept server-side, the
  whole history sent every time. In stateless mode OpenAI returns
  `encrypted_content` on reasoning items by default; xAI returns it
  when `include` asks (grok-4.7 always does), so xAI's requests ask.
- Each reasoning item becomes a reasoning part holding the item
  verbatim as `native`; its readable text is the item's `content`
  (xAI's `reasoning_text`) or its `summary` (OpenAI's, when
  `reasoning.summary` is asked for through `provider_options`).
- Each `message` and `function_call` item becomes a text or tool-call
  part whose `Signature` is the item itself, as JSON, so it goes back
  as the same item, with its id and its `phase` (commentary or final
  answer), which OpenAI asks integrations to preserve.
- Everything goes back only to the provider AND model that wrote it.
  To any other model, text goes as plain assistant text and a call as
  a bare `function_call`; reasoning goes nowhere, and is never text.
- The system prompt is `instructions`; tool results are
  `function_call_output` items; `max_tokens` is `max_output_tokens`;
  JSON mode is `text.format`. There is no `stop`, `seed` or top-level
  `logprobs`, so a request that asks for them is refused (in
  `Transport.check_supported`), never silently changed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.providers import messages as msg
from mechbench_compute.providers.base import AdapterResponse, EmptyReply, Usage

API = "responses"

#: Models whose default is the Responses API, by provider and model
#: prefix. OpenAI's reasoning guide: "Chat Completions does not support
#: function calling with GPT-6 Astra." Every other model keeps Chat
#: Completions unless a request asks.
RESPONSES_BY_DEFAULT: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-6-astra",),
}

#: What each provider needs in `include` for reasoning to come back
#: encrypted when nothing is stored. OpenAI returns it by default in
#: stateless mode and keeps the value only "for compatibility".
INCLUDE: dict[str, tuple[str, ...]] = {
    "xai": ("reasoning.encrypted_content",),
}


def default_api(provider: str, model: str) -> str | None:
    """The API a request to `model` uses when it names none."""
    prefixes = RESPONSES_BY_DEFAULT.get(provider, ())
    return API if any(model.startswith(p) for p in prefixes) else None


# --- requests ---------------------------------------------------------------

def _item_json(item: Mapping[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


def _own_item(sig: msg.Signature | None, kind: str, *, provider: str,
              model: str) -> dict[str, Any] | None:
    """The output item a signature carries, when it is this provider and
    model's own and of the kind expected; otherwise None."""
    if sig is None or not msg.is_replayable(sig.provider, sig.model,
                                            provider=provider, model=model):
        return None
    try:
        item = json.loads(sig.value)
    except (TypeError, ValueError):
        return None
    return item if isinstance(item, dict) and item.get("type") == kind else None


def _message_text(item: Mapping[str, Any]) -> str:
    return "".join(str(c.get("text") or "") for c in item.get("content") or []
                   if isinstance(c, Mapping) and c.get("type") == "output_text")


def _assistant_items(m: msg.Message, provider: str, model: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in m.content:
        if isinstance(p, msg.ReasoningPart):
            # Only a reasoning item this model wrote goes back, verbatim;
            # a DeepSeek field or an Anthropic block never becomes one.
            if (p.native.get("type") == "reasoning"
                    and msg.is_replayable(p.provider, p.model, provider=provider,
                                          model=model)):
                out.append(dict(p.native))
        elif isinstance(p, msg.TextPart):
            item = _own_item(p.signature, "message", provider=provider, model=model)
            if item is not None and _message_text(item) == p.text:
                out.append(item)
            elif p.text:
                out.append({"role": "assistant", "content": p.text})
        elif isinstance(p, msg.ToolCallPart):
            item = _own_item(p.signature, "function_call", provider=provider, model=model)
            if item is not None and item.get("call_id") == p.id and item.get("name") == p.name:
                out.append(item)
            else:
                # No item id: OpenAI refuses a `function_call` item that
                # names its id without the reasoning item it came with.
                out.append({"type": "function_call", "call_id": p.id, "name": p.name,
                            "arguments": json.dumps(dict(p.arguments), sort_keys=True)})
    return out


def input_items(req: msg.ChatRequest, provider: str) -> list[dict[str, Any]]:
    """The conversation as Responses API input items, in order."""
    out: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role == "assistant":
            out.extend(_assistant_items(m, provider, req.model))
            continue
        results = [p for p in m.content if isinstance(p, msg.ToolResultPart)]
        for r in results:
            out.append({"type": "function_call_output", "call_id": r.tool_call_id,
                        "output": r.content})
        text = msg.join_text(m.content)
        if text or not results:
            out.append({"role": "user", "content": text})
    return out


def body(req: msg.ChatRequest, provider: str) -> dict[str, Any]:
    out: dict[str, Any] = {"model": req.model, "input": input_items(req, provider),
                           "max_output_tokens": int(req.max_tokens), "store": False}
    if req.system:
        out["instructions"] = req.system
    if req.tools:
        out["tools"] = [{"type": "function", "name": t.name,
                         "description": t.description,
                         "parameters": dict(t.input_schema)}
                        for t in req.tools]
    if req.tool_choice is not None:
        out["tool_choice"] = (dict(req.tool_choice)
                              if isinstance(req.tool_choice, Mapping)
                              else req.tool_choice)
    if req.temperature is not None:
        out["temperature"] = req.temperature
    if req.top_p is not None:
        out["top_p"] = req.top_p
    if req.json_mode:
        out["text"] = {"format": {"type": "json_object"}}
    if provider in INCLUDE:
        out["include"] = list(INCLUDE[provider])
    out.update(req.options_for(provider))
    return out


# --- responses --------------------------------------------------------------

def read_reasoning(item: Mapping[str, Any], provider: str, model: str) -> msg.ReasoningPart:
    """A reasoning item as a reasoning part: readable text when the item
    has any, the item itself as `native`."""
    def texts(key: str) -> list[str]:
        return [str(c.get("text") or "") for c in item.get(key) or []
                if isinstance(c, Mapping) and c.get("text")]

    text = "".join(texts("content")) or "\n\n".join(texts("summary"))
    return msg.ReasoningPart(text=text, redacted=not text, provider=provider,
                             model=model, native=dict(item))


def _signature(item: Mapping[str, Any], provider: str, model: str) -> msg.Signature:
    return msg.Signature(provider=provider, model=model, value=_item_json(item))


def read_empty(provider: str, *, status: str, reason: str, refusal: str,
               unmapped: Sequence[str], reasoned: bool, usage: Usage,
               max_tokens: int, error: Any = None) -> EmptyReply:
    """Why a response with no prose and no function call came back so."""
    said = f"status {status}" + (f": {reason}" if reason else "")
    if reason == "content_filter" or refusal:
        told = f": {refusal[:200]}" if refusal else ""
        return EmptyReply("filtered", (
            f"{provider} returned no content ({said}"
            f"{', with a refusal' if refusal else ''}){told}. The provider "
            "withheld the reply; the request, not the budget, is what to change."))
    if unmapped and not (reasoned and reason == "max_output_tokens"):
        return EmptyReply("unmapped", (
            f"{provider} answered only with output items this adapter does "
            f"not read ({', '.join(sorted(set(unmapped)))}; {said})."))
    if reasoned or usage.reasoning_tokens:
        spent = (f"{usage.reasoning_tokens} of {usage.output_tokens}"
                 if usage.reasoning_tokens else f"{usage.output_tokens}")
        return EmptyReply("reasoning", (
            f"{provider}: {spent} output tokens went to reasoning "
            f"(max_output_tokens {max_tokens}) and no reply followed ({said}). "
            "Raise max_tokens so the reply has room after the reasoning, or "
            f'lower the reasoning effort with provider_options: {{"{provider}": '
            '{"reasoning": {"effort": "low"}}}.'))
    extra = f" ({error})" if error else ""
    return EmptyReply("no_content", (
        f"{provider} returned no content and no function call ({said}, "
        f"{usage.output_tokens} output tokens){extra}."))


def read_response(data: Mapping[str, Any], req: msg.ChatRequest, *, provider: str,
                  headers: Mapping[str, str] | None = None) -> AdapterResponse:
    """A Responses API body as canonical parts, in output order. Pure: a
    cassette that kept the body maps it again through this on replay."""
    parts: list[msg.Part] = []
    refusal = ""
    unmapped: list[str] = []
    has_text = has_call = reasoned = False
    for item in data.get("output") or []:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "reasoning":
            parts.append(read_reasoning(item, provider, req.model))
            reasoned = True
        elif kind == "message":
            text = _message_text(item)
            refusal = refusal or "".join(
                str(c.get("refusal") or "") for c in item.get("content") or []
                if isinstance(c, Mapping) and c.get("type") == "refusal")
            if text:
                parts.append(msg.TextPart(text, signature=_signature(item, provider,
                                                                     req.model)))
                has_text = True
        elif kind == "function_call":
            raw = item.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                # Invalid JSON from the model is a fact about the run.
                args = {"$raw": raw}
            parts.append(msg.ToolCallPart(
                id=str(item.get("call_id", "")), name=str(item.get("name", "")),
                arguments=args, signature=_signature(item, provider, req.model)))
            has_call = True
        else:
            unmapped.append(str(kind))
    u = data.get("usage") or {}
    usage = Usage(
        input_tokens=int(u.get("input_tokens", 0) or 0),
        output_tokens=int(u.get("output_tokens", 0) or 0),
        cache_read_tokens=int((u.get("input_tokens_details") or {})
                              .get("cached_tokens", 0) or 0),
        reasoning_tokens=int((u.get("output_tokens_details") or {})
                             .get("reasoning_tokens", 0) or 0),
    )
    status = str(data.get("status") or "completed")
    reason = str((data.get("incomplete_details") or {}).get("reason") or "")
    empty = None
    if not has_text and not has_call:
        empty = read_empty(provider, status=status, reason=reason, refusal=refusal,
                           unmapped=unmapped, reasoned=reasoned, usage=usage,
                           max_tokens=int(req.max_tokens), error=data.get("error"))
    return AdapterResponse(
        parts=tuple(parts), stop_reason=reason or status, usage=usage,
        model_version=str(data.get("model") or req.model),
        response_id=str(data.get("id") or ""), headers=dict(headers or {}),
        raw=data, empty=empty)
