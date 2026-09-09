"""The chat-completions shape (task 000337).

One adapter serves OpenAI, xAI, Fireworks, Together, Groq, DeepSeek,
Mistral, OpenRouter, and a llama.cpp or vLLM server on localhost —
they all speak `/chat/completions`. What differs is the base URL, the
model id, and which capabilities are real, so those are parameters,
not subclasses.

Mapping notes that matter:

- The system prompt becomes a leading `system` message here (it is a
  field in the canonical model, so the hash of a conversation does not
  depend on which adapter answers it).
- Tool calls ride on the assistant message as `tool_calls` with
  JSON-STRING arguments; tool results are separate `role: "tool"`
  messages. The canonical model keeps both as parts of the turn they
  belong to, so this is where they split and rejoin.
- `logprobs` is `top_logprobs`, capped at 20 on OpenAI (more on some
  self-hosted servers) — the cap is a capability, declared per host.
"""

from __future__ import annotations

import json
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
from mechbench_compute.providers.errors import AuthError

#: provider name -> (base URL, capabilities). A host not listed here is
#: reachable as `openai-compatible` with an explicit base_url.
HOSTS: dict[str, tuple[str, Capabilities]] = {
    "openai": ("https://api.openai.com/v1", Capabilities(
        chat=True, complete=False, count_tokens=None, tools=True,
        json_mode=True, seed=True, logprobs=20, cache_control=False,
        batch=True, embed=True, streaming=False, models=True)),
    "xai": ("https://api.x.ai/v1", Capabilities(
        chat=True, tools=True, json_mode=True, seed=True, logprobs=8,
        batch=False, embed=False, models=True)),
    "fireworks": ("https://api.fireworks.ai/inference/v1", Capabilities(
        chat=True, complete=True, tools=True, json_mode=True, seed=True,
        logprobs=5, batch=False, embed=True, models=True)),
    "openai-compatible": ("", Capabilities(
        chat=True, tools=True, json_mode=True, seed=True, logprobs=5,
        models=True)),
}


def _messages(req: msg.ChatRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    for m in req.messages:
        text = "".join(p.text for p in m.content if isinstance(p, msg.TextPart))
        calls = [p for p in m.content if isinstance(p, msg.ToolCallPart)]
        results = [p for p in m.content if isinstance(p, msg.ToolResultPart)]
        if m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name,
                                  "arguments": json.dumps(dict(c.arguments),
                                                          sort_keys=True)}}
                    for c in calls]
            out.append(entry)
            continue
        # A user turn may carry tool results, which this API models as
        # their own messages BEFORE the user's text.
        for r in results:
            out.append({"role": "tool", "tool_call_id": r.tool_call_id,
                        "content": r.content})
        if text or not results:
            out.append({"role": "user", "content": text})
    return out


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
        # A local server (llama.cpp, vLLM, Ollama) commonly has no key —
        # that is legitimate, and refusing it would block the cheapest
        # remote node there is.
        self._token = str(token) if token else ""
        self.name = provider
        self.capabilities = capabilities or caps
        self._base = str(base).rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self._token}"} if self._token else {}

    def _body(self, req: msg.ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"model": req.model, "messages": _messages(req),
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
        resp = http.post_json(f"{self._base}/chat/completions", headers=self._headers(),
                              payload=self._body(req), timeout=self._timeout,
                              secrets=(self._token,))
        data = resp.body or {}
        choice = (data.get("choices") or [{}])[0]
        m = choice.get("message") or {}
        parts: list[msg.Part] = []
        if m.get("content"):
            parts.append(msg.TextPart(str(m["content"])))
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                # A model that emitted invalid JSON is a fact about the
                # run, not a crash: keep it verbatim for the reader.
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
        return AdapterResponse(
            parts=tuple(parts),
            stop_reason=str(choice.get("finish_reason") or "stop"),
            usage=usage, model_version=str(data.get("model") or req.model),
            response_id=str(data.get("id") or ""), headers=resp.headers,
            logprobs=(choice.get("logprobs") or None), raw=data)

    def models(self) -> list[str]:
        resp = http.get_json(f"{self._base}/models", headers=self._headers(),
                             timeout=30.0, secrets=(self._token,))
        return [str(m.get("id")) for m in (resp.body or {}).get("data", [])]
