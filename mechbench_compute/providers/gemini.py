"""Google's generateContent API.

The shape differs more than the others: turns are `contents` with the
assistant called `model`, tools are `functionDeclarations`, sampling
knobs live under `generationConfig`, and a tool result is a
`functionResponse` keyed by the function's NAME rather than the call's
id — so this adapter resolves ids back to names from the conversation
it was handed, which is the one place the canonical model needs
translating rather than renaming.

Reasoning comes two ways. A part marked `thought: true` is a thought
summary: a reasoning part, never reply text. And any part — a
`functionCall`, the last text part — may carry a `thoughtSignature`,
which must go back on that same part: a function call in the current
turn whose signature is missing is refused. (With parallel calls only
the first carries one.) The signature is kept on the canonical part it
rode on, and both kinds go back only to the model that issued them.
"""

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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

CAPABILITIES = Capabilities(
    chat=True, complete=False, count_tokens="exact", tools=True,
    json_mode=True, seed=False, logprobs=5, cache_control=True,
    batch=True, embed=True, streaming=False, models=True,
)


#: The finish reasons that mean the provider withheld the content.
SAFETY_FINISH_REASONS = frozenset({
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
})


def _contents(req: msg.ChatRequest) -> list[dict[str, Any]]:
    names: dict[str, str] = {}
    for m in req.messages:
        for p in m.content:
            if isinstance(p, msg.ToolCallPart):
                names[p.id] = p.name
    out: list[dict[str, Any]] = []
    for m in req.messages:
        parts: list[dict[str, Any]] = []
        for p in m.content:
            if isinstance(p, msg.ReasoningPart):
                # Another model's thoughts are dropped, never turned
                # into text.
                if p.native and msg.is_replayable(p.provider, p.model,
                                                  provider="gemini", model=req.model):
                    parts.append(dict(p.native))
                continue
            if isinstance(p, msg.TextPart):
                parts.append({"text": p.text})
            elif isinstance(p, msg.ToolCallPart):
                parts.append({"functionCall": {"name": p.name,
                                               "args": dict(p.arguments)}})
            else:
                parts.append({"functionResponse": {
                    "name": names.get(p.tool_call_id, p.tool_call_id),
                    "response": {"content": p.content,
                                 **({"error": True} if p.is_error else {})}}})
            sig = getattr(p, "signature", None)
            if sig is not None and msg.is_replayable(sig.provider, sig.model,
                                                     provider="gemini", model=req.model):
                parts[-1]["thoughtSignature"] = sig.value
        out.append({"role": "model" if m.role == "assistant" else "user",
                    "parts": parts})
    return out


def read_empty(data: Mapping[str, Any], cand: Mapping[str, Any], *,
               usage: Usage, max_tokens: int) -> EmptyReply:
    """Why a response with no reply text and no function call came back
    that way: a blocked prompt, no candidates, or a safety-class finish
    reason; thoughts that used the allowance; or nothing it says."""
    block = (data.get("promptFeedback") or {}).get("blockReason")
    finish = str(cand.get("finishReason") or "")
    if block:
        return EmptyReply("filtered", (
            f"gemini blocked the prompt (promptFeedback.blockReason {block}) "
            "and returned no candidates. The request, not the budget, is "
            "what to change."))
    if not data.get("candidates"):
        return EmptyReply("filtered", (
            "gemini returned no candidates, which it does only when "
            "something about the prompt was refused."))
    if finish in SAFETY_FINISH_REASONS:
        return EmptyReply("filtered", (
            f"gemini withheld the reply (finishReason {finish}). The "
            "request, not the budget, is what to change."))
    if usage.reasoning_tokens:
        return EmptyReply("reasoning", (
            f"gemini: {usage.reasoning_tokens} of {usage.output_tokens} output "
            f"tokens went to thoughts (maxOutputTokens {max_tokens}) and no "
            f"reply text followed (finishReason {finish or 'STOP'}). Raise "
            "max_tokens so the reply has room after the thinking, or lower "
            "the thinking budget with provider_options: "
            '{"gemini": {"generationConfig": {"thinkingConfig": '
            '{"thinkingBudget": 0}}}} (a model that takes a level rather '
            'than a budget takes {"thinkingLevel": "low"}).'))
    return EmptyReply("no_content", (
        f"gemini returned no text and no function call (finishReason "
        f"{finish or 'STOP'}, {usage.output_tokens} output tokens)."))


def read_response(data: Mapping[str, Any], req: msg.ChatRequest, *,
                  headers: Mapping[str, str] | None = None) -> AdapterResponse:
    """A generateContent response body as canonical parts. Pure: a
    cassette that kept the body maps it again through this on replay."""
    cand = (data.get("candidates") or [{}])[0]
    parts: list[msg.Part] = []
    prose = False
    for i, p in enumerate((cand.get("content") or {}).get("parts") or []):
        if p.get("thought"):
            parts.append(msg.ReasoningPart(
                text=str(p.get("text") or ""), redacted=not p.get("text"),
                provider="gemini", model=req.model, native=dict(p)))
            continue
        sig = (msg.Signature("gemini", req.model, str(p["thoughtSignature"]))
               if p.get("thoughtSignature") else None)
        if "text" in p:
            parts.append(msg.TextPart(str(p["text"]), signature=sig))
            prose = prose or bool(p["text"])
        elif "functionCall" in p:
            fc = p["functionCall"]
            parts.append(msg.ToolCallPart(
                # Gemini does not issue call ids; the index is stable
                # within a response and correlates the result we send
                # back (which is keyed by name anyway).
                id=f"{cand.get('index', 0)}-{i}", name=str(fc.get("name", "")),
                arguments=dict(fc.get("args") or {}), signature=sig))
    u = data.get("usageMetadata") or {}
    thoughts = int(u.get("thoughtsTokenCount", 0) or 0)
    usage = Usage(
        input_tokens=int(u.get("promptTokenCount", 0)),
        # Thoughts are billed as output and count against
        # maxOutputTokens, but `candidatesTokenCount` leaves them out.
        output_tokens=int(u.get("candidatesTokenCount", 0) or 0) + thoughts,
        cache_read_tokens=int(u.get("cachedContentTokenCount", 0) or 0),
        reasoning_tokens=thoughts,
    )
    stop_reason = str(cand.get("finishReason") or "STOP").lower()
    empty = None
    if not prose and not any(isinstance(p, msg.ToolCallPart) for p in parts):
        empty = read_empty(data, cand, usage=usage, max_tokens=int(req.max_tokens))
    return AdapterResponse(
        parts=tuple(parts),
        stop_reason=stop_reason,
        usage=usage,
        model_version=str(data.get("modelVersion") or req.model),
        response_id=str(data.get("responseId") or ""), headers=dict(headers or {}),
        logprobs=cand.get("logprobsResult"), raw=data, empty=empty)


class GeminiTransport(Transport):
    name = "gemini"
    capabilities = CAPABILITIES

    def __init__(self, credential: Mapping[str, Any] | str, *,
                 base_url: str | None = None, timeout: float = http.DEFAULT_TIMEOUT,
                 sleep=None, clock=None) -> None:
        super().__init__(sleep=sleep, clock=clock)
        token = credential if isinstance(credential, str) else credential.get("token")
        if not token:
            raise AuthError("gemini: no API key in the delivered credential")
        self._token = str(token)
        base = (base_url or (credential.get("base_url")
                             if isinstance(credential, Mapping) else None)
                or DEFAULT_BASE_URL)
        self._base = str(base).rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        # The key goes in a header, never the query string: a URL ends
        # up in logs and error messages, and this one would carry the
        # account with it.
        return {"x-goog-api-key": self._token}

    def _body(self, req: msg.ChatRequest) -> dict[str, Any]:
        cfg: dict[str, Any] = {"maxOutputTokens": int(req.max_tokens)}
        if req.temperature is not None:
            cfg["temperature"] = req.temperature
        if req.top_p is not None:
            cfg["topP"] = req.top_p
        if req.stop:
            cfg["stopSequences"] = list(req.stop)
        if req.json_mode:
            cfg["responseMimeType"] = "application/json"
        if req.logprobs is not None:
            cfg["responseLogprobs"] = True
            cfg["logprobs"] = int(req.logprobs)
        body: dict[str, Any] = {"contents": _contents(req), "generationConfig": cfg}
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        if req.tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description,
                 "parameters": dict(t.input_schema)} for t in req.tools]}]
        if req.tool_choice is not None:
            mode = (req.tool_choice if isinstance(req.tool_choice, str)
                    else dict(req.tool_choice).get("mode", "AUTO"))
            body["toolConfig"] = {"functionCallingConfig": {"mode": str(mode).upper()}}
        options = dict(req.options_for(self.name))
        # `generationConfig` in the options is merged into the one built
        # here, not swapped for it: replacing it would drop
        # `maxOutputTokens`, and the budget reserved against max_tokens
        # would no longer bound what the call can spend.
        if isinstance(options.get("generationConfig"), Mapping):
            cfg.update(options.pop("generationConfig"))
        body.update(options)
        return body

    def _chat(self, req: msg.ChatRequest, *, on_token=None) -> AdapterResponse:
        url = f"{self._base}/models/{req.model}:generateContent"
        resp = http.post_json(url, headers=self._headers(), payload=self._body(req),
                              timeout=self._timeout, secrets=(self._token,))
        return read_response(resp.body or {}, req, headers=resp.headers)

    def _count_tokens(self, req: msg.ChatRequest) -> int:
        url = f"{self._base}/models/{req.model}:countTokens"
        body = {"contents": _contents(req)}
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        resp = http.post_json(url, headers=self._headers(), payload=body,
                              timeout=min(self._timeout, 60.0),
                              secrets=(self._token,))
        return int((resp.body or {}).get("totalTokens", 0))

    def models(self) -> list[str]:
        resp = http.get_json(f"{self._base}/models", headers=self._headers(),
                             timeout=30.0, secrets=(self._token,))
        return [str(m.get("name", "")).removeprefix("models/")
                for m in (resp.body or {}).get("models", [])]
