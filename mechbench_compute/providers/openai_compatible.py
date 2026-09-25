from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mechbench_compute.providers import http
from mechbench_compute.providers import messages as msg
from mechbench_compute.providers.base import (
    AdapterResponse,
    Capabilities,
    EmptyReply,
    Transport,
    Usage,
)
from mechbench_compute.providers.errors import AuthError

HOSTS: dict[str, tuple[str, Capabilities]] = {
    "openai": ("https://api.openai.com/v1", Capabilities(
        chat=True, complete=False, count_tokens=None, tools=True,
        json_mode=True, seed=True, logprobs=20, cache_control=False,
        batch=True, embed=True, streaming=False, models=True, responses=True, reasoning=True)),
    "xai": ("https://api.x.ai/v1", Capabilities(
        chat=True, tools=True, json_mode=True, seed=True, logprobs=8,
        batch=False, embed=False, models=True, responses=True, reasoning=True)),
    "fireworks": ("https://api.fireworks.ai/inference/v1", Capabilities(
        chat=True, complete=True, tools=True, json_mode=True, seed=True,
        logprobs=5, batch=False, embed=True, models=True, reasoning=True)),
    "deepseek": ("https://api.deepseek.com", Capabilities(
        chat=True, tools=True, json_mode=True, seed=False, logprobs=None,
        batch=False, embed=False, models=True, reasoning=True)),
    "openai-compatible": ("", Capabilities(
        chat=True, tools=True, json_mode=True, seed=True, logprobs=5,
        models=True, reasoning=True)),
}


REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_details")

REPLAYS_REASONING = frozenset({"deepseek", "fireworks", "openai-compatible"})

INLINE_THOUGHT = ("<think>", "</think>")


def split_inline_thought(content: str) -> tuple[str | None, str]:
    open_s, close_s = INLINE_THOUGHT
    head = content.lstrip()
    if head.startswith(open_s):
        body = head[len(open_s):]
        end = body.find(close_s)
        if end == -1:
            return body.strip(), ""
        return body[:end].strip(), body[end + len(close_s):].lstrip()
    end = content.find(close_s)
    if end != -1 and open_s not in content[:end]:
        return content[:end].strip(), content[end + len(close_s):].lstrip()
    return None, content


def read_reasoning(message: Mapping[str, Any], provider: str,
                   model: str) -> msg.ReasoningPart | None:
    native = {k: message[k] for k in REASONING_FIELDS if message.get(k)}
    if not native:
        return None
    text = next((str(native[k]) for k in ("reasoning_content", "reasoning")
                 if isinstance(native.get(k), str)), "")
    details = native.get("reasoning_details")
    if not text and isinstance(details, list):
        text = "".join(str(d.get("text") or d.get("summary") or "")
                       for d in details if isinstance(d, Mapping))
    return msg.ReasoningPart(text=text, redacted=not text, provider=provider,
                             model=model, native=native)


def _messages(req: msg.ChatRequest, provider: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    for m in req.messages:
        text = msg.join_text(m.content)
        reasoning = [p for p in m.content if isinstance(p, msg.ReasoningPart)
                     and p.native and provider in REPLAYS_REASONING
                     and msg.is_replayable(p.provider, p.model, provider=provider,
                                           model=req.model)]
        calls = [p for p in m.content if isinstance(p, msg.ToolCallPart)]
        results = [p for p in m.content if isinstance(p, msg.ToolResultPart)]
        if m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            for r in reasoning:
                entry.update(dict(r.native))
            if calls:
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name,
                                  "arguments": json.dumps(dict(c.arguments),
                                                          sort_keys=True)}}
                    for c in calls]
            out.append(entry)
            continue
        for r in results:
            out.append({"role": "tool", "tool_call_id": r.tool_call_id,
                        "content": r.content})
        if text or not results:
            out.append({"role": "user", "content": text})
    return out


def read_empty(provider: str, message: Mapping[str, Any], *, stop_reason: str,
               usage: Usage, max_tokens: int, reasoned: bool = False) -> EmptyReply:
    refusal = message.get("refusal")
    if stop_reason == "content_filter" or refusal:
        said = f": {str(refusal)[:200]}" if refusal else ""
        return EmptyReply("filtered", (
            f"{provider} returned no content (finish_reason {stop_reason}"
            f"{', with a refusal' if refusal else ''}){said}. The provider "
            "withheld the reply; the request, not the budget, is what to change."))
    if usage.reasoning_tokens or reasoned:
        spent = (f"{usage.reasoning_tokens} of {usage.output_tokens}"
                 if usage.reasoning_tokens else f"{usage.output_tokens}")
        return EmptyReply("reasoning", (
            f"{provider}: {spent} "
            f"completion tokens went to reasoning (max_tokens {max_tokens}) "
            f"and no content followed (finish_reason {stop_reason}). Raise "
            "max_tokens so the reply has room after the reasoning, or lower "
            f'the reasoning effort with provider_options: {{"{provider}": '
            '{"reasoning_effort": "low"}}.'))
    return EmptyReply("no_content", (
        f"{provider} returned no content and no tool call (finish_reason "
        f"{stop_reason}, {usage.output_tokens} completion tokens)."))


def read_response(data: Mapping[str, Any], req: msg.ChatRequest, *,
                  provider: str,
                  headers: Mapping[str, str] | None = None) -> AdapterResponse:
    choice = (data.get("choices") or [{}])[0]
    m = choice.get("message") or {}
    parts: list[msg.Part] = []
    reasoning = read_reasoning(m, provider, req.model)
    content = str(m.get("content") or "")
    if reasoning is None and content:
        inline, content = split_inline_thought(content)
        if inline is not None:
            reasoning = msg.ReasoningPart(text=inline, redacted=not inline,
                                          provider=provider, model=req.model)
    if reasoning is not None:
        parts.append(reasoning)
    if content:
        parts.append(msg.TextPart(content))
    for c in m.get("tool_calls") or []:
        fn = c.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            args = {"$raw": raw}
        parts.append(msg.ToolCallPart(id=str(c.get("id", "")),
                                      name=str(fn.get("name", "")),
                                      arguments=args))
    u = data.get("usage") or {}
    details = u.get("prompt_tokens_details") or {}
    usage = Usage(
        input_tokens=int(u.get("prompt_tokens", 0)),
        output_tokens=int(u.get("completion_tokens", 0)),
        cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
        reasoning_tokens=int((u.get("completion_tokens_details") or {})
                             .get("reasoning_tokens", 0) or 0),
    )
    stop_reason = str(choice.get("finish_reason") or "stop")
    empty = None
    if not content and not m.get("tool_calls"):
        empty = read_empty(provider, m, stop_reason=stop_reason, usage=usage,
                           max_tokens=int(req.max_tokens),
                           reasoned=reasoning is not None)
    return AdapterResponse(
        parts=tuple(parts), stop_reason=stop_reason,
        usage=usage, model_version=str(data.get("model") or req.model),
        response_id=str(data.get("id") or ""), headers=dict(headers or {}),
        logprobs=(choice.get("logprobs") or None), raw=data, empty=empty)


class OpenAICompatibleTransport(Transport):
    def __init__(self, credential: Mapping[str, Any] | str, *, provider: str = "openai",
                 base_url: str | None = None, timeout: float = http.DEFAULT_TIMEOUT,
                 capabilities: Capabilities | None = None, sleep=None, clock=None) -> None:
        super().__init__(sleep=sleep, clock=clock)
        default_base, caps = HOSTS.get(provider, HOSTS["openai-compatible"])
        cred = {"token": credential} if isinstance(credential, str) else dict(credential or {})
        base = base_url or cred.get("base_url") or default_base
        if not base:
            raise AuthError(
                f"{provider}: an openai-compatible endpoint needs a base_url "
                "(the credential carries {token, base_url})")
        token = cred.get("token")
        self._token = str(token) if token else ""
        self.name = provider
        self.capabilities = capabilities or caps
        self._base = str(base).rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self._token}"} if self._token else {}

    def _body(self, req: msg.ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"model": req.model,
                                "messages": _messages(req, self.name),
                                "max_tokens": int(req.max_tokens)}
        if req.tools:
            body["tools"] = [{"type": "function",
                              "function": {"name": t.name,
                                           "description": t.description,
                                           "parameters": dict(t.input_schema)}}
                             for t in req.tools]
        if req.tool_choice is not None:
            body["tool_choice"] = (dict(req.tool_choice)
                                   if isinstance(req.tool_choice, Mapping)
                                   else req.tool_choice)
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop"] = list(req.stop)
        if req.seed is not None:
            body["seed"] = int(req.seed)
        if req.json_mode:
            body["response_format"] = {"type": "json_object"}
        if req.logprobs is not None:
            body["logprobs"] = True
            body["top_logprobs"] = int(req.logprobs)
        body.update(req.options_for(self.name))
        return body

    def _chat(self, req: msg.ChatRequest, *, on_token=None) -> AdapterResponse:
        if req.api == "responses":
            from mechbench_compute.providers import openai_responses

            resp = http.post_json(f"{self._base}/responses", headers=self._headers(),
                                  payload=openai_responses.body(req, self.name),
                                  timeout=self._timeout, secrets=(self._token,))
            return openai_responses.read_response(resp.body or {}, req, provider=self.name,
                                                  headers=resp.headers)
        resp = http.post_json(f"{self._base}/chat/completions", headers=self._headers(),
                              payload=self._body(req), timeout=self._timeout,
                              secrets=(self._token,))
        return read_response(resp.body or {}, req, provider=self.name,
                             headers=resp.headers)

    def models(self) -> list[str]:
        resp = http.get_json(f"{self._base}/models", headers=self._headers(),
                             timeout=30.0, secrets=(self._token,))
        return [str(m.get("id")) for m in (resp.body or {}).get("data", [])]
