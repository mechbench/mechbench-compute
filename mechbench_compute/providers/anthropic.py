"""The Anthropic Messages API (task 000337).

The canonical model was shaped after this one (system as a field,
content parts, tool_use / tool_result), so the mapping is nearly
transparent — which is the point of choosing a shape that a provider
already agrees with rather than a lowest common denominator.

What is NOT here, by capability: logprobs (the API offers none) and
`seed` (sampling is not reproducible), so a protocol that asks for
either is refused by name before the job starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.providers import http
from mechbench_compute.providers import messages as msg
from mechbench_compute.providers.base import (
    AdapterResponse,
    Capabilities,
    Transport,
    Usage,
)
from mechbench_compute.providers.errors import AuthError, ProviderError

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"

CAPABILITIES = Capabilities(
    chat=True, complete=False, count_tokens="exact", tools=True,
    json_mode=False, seed=False, logprobs=None, cache_control=True,
    batch=True, embed=False, streaming=False, models=True,
)


def _content(m: msg.Message) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in m.content:
        if isinstance(p, msg.TextPart):
            out.append({"type": "text", "text": p.text})
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
            "messages": [{"role": m.role, "content": _content(m)} for m in req.messages],
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
        # Passthrough LAST: `provider_options.anthropic` is how a caller
        # reaches cache_control, thinking budgets, service tiers and
        # anything this module has not grown a field for.
        body.update(req.options_for(self.name))
        return body

    def _chat(self, req: msg.ChatRequest, *, on_token=None) -> AdapterResponse:
        resp = http.post_json(f"{self._base}/v1/messages", headers=self._headers(),
                              payload=self._body(req), timeout=self._timeout,
                              secrets=(self._token,))
        data = resp.body or {}
        parts: list[msg.Part] = []
        unmapped: list[str] = []
        for block in data.get("content") or []:
            kind = block.get("type")
            if kind == "text":
                parts.append(msg.TextPart(block.get("text", "")))
            elif kind == "tool_use":
                parts.append(msg.ToolCallPart(id=str(block.get("id", "")),
                                              name=str(block.get("name", "")),
                                              arguments=dict(block.get("input") or {})))
            elif kind in ("thinking", "redacted_thinking"):
                # Reasoning is content, not text: keep it addressable
                # rather than dropping it, so a turn that "came back
                # empty" can be explained.
                parts.append(msg.TextPart(str(block.get("thinking", "")))
                             if block.get("thinking") else msg.TextPart(""))
                unmapped.append(kind)
            else:
                # An unknown block type must never vanish silently: an
                # empty completion beside 250 output tokens is how
                # experiment 024 found this.
                unmapped.append(str(kind))
        u = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(u.get("input_tokens", 0))
            + int(u.get("cache_read_input_tokens", 0) or 0)
            + int(u.get("cache_creation_input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0)),
            cache_read_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
        )
        if unmapped and not any(isinstance(p, msg.TextPart) and p.text
                                for p in parts):
            raise ProviderError(
                f"anthropic returned {usage.output_tokens} output tokens but no "
                f"text: content block type(s) {', '.join(sorted(set(unmapped)))}. "
                "The adapter does not map these — the completion is not empty, "
                "it is unreadable here.")
        return AdapterResponse(
            parts=tuple(parts), stop_reason=str(data.get("stop_reason") or "end_turn"),
            usage=usage, model_version=str(data.get("model") or req.model),
            response_id=str(data.get("id") or ""), headers=resp.headers, raw=data)

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
