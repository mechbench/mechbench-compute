from __future__ import annotations

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

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"

CAPABILITIES = Capabilities(
    chat=True, complete=False, count_tokens="exact", tools=True,
    json_mode=False, seed=False, logprobs=None, cache_control=True,
    batch=True, embed=False, streaming=False, models=True, reasoning=True,
)

REASONING_BLOCKS = frozenset({"thinking", "redacted_thinking"})


def _content(m: msg.Message, model: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in m.content:
        if isinstance(p, msg.TextPart):
            out.append({"type": "text", "text": p.text})
        elif isinstance(p, msg.ReasoningPart):
            if p.native and msg.is_replayable(p.provider, p.model, provider="anthropic",
                                              model=model, model_bound=False):
                out.append(dict(p.native))
        elif isinstance(p, msg.ToolCallPart):
            out.append({"type": "tool_use", "id": p.id, "name": p.name,
                        "input": dict(p.arguments)})
        else:
            entry: dict[str, Any] = {"type": "tool_result",
                                     "tool_use_id": p.tool_call_id,
                                     "content": p.content}
            if p.is_error:
                entry["is_error"] = True
            out.append(entry)
    return out


def read_block(block: Mapping[str, Any], model: str) -> msg.ReasoningPart:
    text = str(block.get("thinking") or "")
    return msg.ReasoningPart(
        text=text, redacted=block.get("type") == "redacted_thinking" or not text,
        provider="anthropic", model=model, native=dict(block))


def read_response(data: Mapping[str, Any], req: msg.ChatRequest, *,
                  headers: Mapping[str, str] | None = None) -> AdapterResponse:
    parts: list[msg.Part] = []
    unmapped: list[str] = []
    prose = False
    for block in data.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(msg.TextPart(block.get("text", "")))
            prose = prose or bool(block.get("text"))
        elif kind == "tool_use":
            parts.append(msg.ToolCallPart(id=str(block.get("id", "")),
                                          name=str(block.get("name", "")),
                                          arguments=dict(block.get("input") or {})))
        elif kind in REASONING_BLOCKS:
            parts.append(read_block(block, req.model))
            unmapped.append(kind)
        else:
            unmapped.append(str(kind))
    u = data.get("usage") or {}
    usage = Usage(
        input_tokens=int(u.get("input_tokens", 0))
        + int(u.get("cache_read_input_tokens", 0) or 0)
        + int(u.get("cache_creation_input_tokens", 0) or 0),
        output_tokens=int(u.get("output_tokens", 0)),
        cache_read_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
        reasoning_tokens=int((u.get("output_tokens_details") or {})
                             .get("thinking_tokens", 0) or 0),
        cache_write_1h_tokens=int((u.get("cache_creation") or {})
                                  .get("ephemeral_1h_input_tokens", 0) or 0),
    )
    stop_reason = str(data.get("stop_reason") or "end_turn")
    empty = None
    if not prose and not any(isinstance(p, msg.ToolCallPart) for p in parts):
        empty = read_empty(
            unmapped, stop_reason=stop_reason, usage=usage,
            max_tokens=int(req.max_tokens),
            reasoning_text=any(isinstance(p, msg.ReasoningPart) and p.text
                               for p in parts))
    return AdapterResponse(
        parts=tuple(parts), stop_reason=stop_reason,
        usage=usage, model_version=str(data.get("model") or req.model),
        response_id=str(data.get("id") or ""), headers=dict(headers or {}),
        raw=data, empty=empty)


def read_empty(unmapped: list[str], *, stop_reason: str, usage: Usage,
               max_tokens: int, reasoning_text: bool = False) -> EmptyReply:
    kinds = ", ".join(sorted(set(unmapped)))
    if stop_reason == "refusal":
        return EmptyReply("filtered", (
            f"anthropic stopped with stop_reason refusal after "
            f"{usage.output_tokens} output tokens and returned no prose: the "
            "model declined the request."))
    if unmapped and set(unmapped) <= REASONING_BLOCKS:
        return EmptyReply("reasoning", (
            f"anthropic: {usage.output_tokens} of {max_tokens} "
            "output tokens (max_tokens) went to reasoning and no prose "
            f"followed (content block type(s) {kinds}). "
            + ("The completion holds reasoning only, kept as the item's "
               "`reasoning` and not as a reply. " if reasoning_text else
               "The completion is not empty: it holds reasoning whose text "
               "the API did not return. ")
            + "Raise max_tokens so the reply has room after "
            "the reasoning, or turn reasoning off with provider_options: "
            '{"anthropic": {"thinking": {"type": "disabled"}}}.'))
    if unmapped:
        return EmptyReply("unmapped", (
            f"anthropic returned {usage.output_tokens} output tokens but no "
            f"text: content block type(s) {kinds}. "
            "The adapter does not map these — the completion is not empty, "
            "it is unreadable here."))
    return EmptyReply("no_content", (
        f"anthropic returned no text and no tool call (stop_reason "
        f"{stop_reason}, {usage.output_tokens} output tokens)."))


class AnthropicTransport(Transport):
    name = "anthropic"
    capabilities = CAPABILITIES

    def __init__(self, credential: Mapping[str, Any] | str, *,
                 base_url: str | None = None, timeout: float = http.DEFAULT_TIMEOUT,
                 sleep=None, clock=None) -> None:
        super().__init__(sleep=sleep, clock=clock)
        token = credential if isinstance(credential, str) else credential.get("token")
        if not token:
            raise AuthError("anthropic: no API token in the delivered credential")
        self._token = str(token)
        base = (base_url or (credential.get("base_url")
                             if isinstance(credential, Mapping) else None)
                or DEFAULT_BASE_URL)
        self._base = str(base).rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._token, "anthropic-version": API_VERSION}

    def _body(self, req: msg.ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "max_tokens": int(req.max_tokens),
            "messages": [{"role": m.role, "content": _content(m, req.model)}
                         for m in req.messages],
        }
        if req.system:
            body["system"] = req.system
        if req.tools:
            body["tools"] = [{"name": t.name, "description": t.description,
                              "input_schema": dict(t.input_schema)} for t in req.tools]
        if req.tool_choice is not None:
            body["tool_choice"] = (dict(req.tool_choice)
                                   if isinstance(req.tool_choice, Mapping)
                                   else {"type": str(req.tool_choice)})
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop_sequences"] = list(req.stop)
        body.update(req.options_for(self.name))
        return body

    def _chat(self, req: msg.ChatRequest, *, on_token=None) -> AdapterResponse:
        resp = http.post_json(f"{self._base}/v1/messages", headers=self._headers(),
                              payload=self._body(req), timeout=self._timeout,
                              secrets=(self._token,))
        return read_response(resp.body or {}, req, headers=resp.headers)

    def _count_tokens(self, req: msg.ChatRequest) -> int:
        body = self._body(req)
        body.pop("max_tokens", None)
        resp = http.post_json(f"{self._base}/v1/messages/count_tokens",
                              headers=self._headers(), payload=body,
                              timeout=min(self._timeout, 60.0),
                              secrets=(self._token,))
        return int((resp.body or {}).get("input_tokens", 0))

    def models(self) -> list[str]:
        resp = http.get_json(f"{self._base}/v1/models", headers=self._headers(),
                             timeout=30.0, secrets=(self._token,))
        return [str(m.get("id")) for m in (resp.body or {}).get("data", [])]
