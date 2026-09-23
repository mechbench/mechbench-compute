from __future__ import annotations

#: Each provider's stop reasons, as the adapter records them in
#: `metadata.call.stop_reason`, onto the endings of `ENDINGS`, each
#: table as the provider's API reference lists its words. A word not
#: listed is `other`, and stays verbatim on the call record.
#:
#: Anthropic Messages `stop_reason`: `end_turn`, `max_tokens`,
#: `stop_sequence`, `tool_use`, `pause_turn`, `refusal`,
#: `model_context_window_exceeded` (the reply filled the context window
#: before `max_tokens`: cut off all the same).
_ANTHROPIC = {
    "end_turn": "end", "stop_sequence": "stop", "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens", "tool_use": "tool_call",
    "refusal": "filtered",
}
#: OpenAI Chat Completions `finish_reason`: `stop` (a natural stop OR a
#: stop sequence — the API does not say which, so `end`), `length`,
#: `tool_calls`, `content_filter`, `function_call` (deprecated). The
#: Responses API records `incomplete_details.reason` when there is one
#: (`max_output_tokens`, `content_filter`, `max_messages`, `steered`),
#: else the `status` (`completed`, `incomplete`, `failed`, `cancelled`,
#: `in_progress`, `queued`).
_OPENAI = {
    "stop": "end", "length": "max_tokens", "tool_calls": "tool_call",
    "function_call": "tool_call", "content_filter": "filtered",
    "completed": "end", "max_output_tokens": "max_tokens",
}
#: xAI Chat Completions: `stop` (a model-defined or user-supplied stop
#: sequence), `length`, `end_turn`, and the OpenAI-compatible
#: `tool_calls` / `content_filter`; its Responses API has `completed`,
#: `in_progress`, `incomplete`, read as OpenAI's.
_XAI = {**_OPENAI, "end_turn": "end"}
#: DeepSeek `finish_reason`: `stop` (a natural stop or a stop sequence),
#: `length`, `content_filter`, `tool_calls`, `insufficient_system_resource`
#: and `aborted` (interrupted: `other`).
_DEEPSEEK = {
    "stop": "end", "length": "max_tokens", "tool_calls": "tool_call",
    "content_filter": "filtered",
}
#: Gemini `finishReason`, lowercased by the adapter: `STOP` (a natural
#: stop or a stop sequence, so `end`), `MAX_TOKENS`, the filters
#: (`SAFETY`, `RECITATION`, `LANGUAGE`, `BLOCKLIST`, `PROHIBITED_CONTENT`,
#: `SPII`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_RECITATION`,
#: `ESCALATION`, `PUP_LIMITED_DISABLED`), and the rest `other` (`OTHER`,
#: `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `TOO_MANY_TOOL_CALLS`,
#: `MISSING_THOUGHT_SIGNATURE`, `MALFORMED_RESPONSE`, `IMAGE_OTHER`,
#: `NO_IMAGE`, `FINISH_REASON_UNSPECIFIED`). Gemini says `STOP` for a
#: function call too; the reply's parts tell the two apart.
_GEMINI = {
    "stop": "end", "max_tokens": "max_tokens",
    **dict.fromkeys(("safety", "recitation", "language", "blocklist",
                     "prohibited_content", "spii", "image_safety",
                     "image_prohibited_content", "image_recitation",
                     "escalation", "pup_limited_disabled"), "filtered"),
}
_BY_PROVIDER = {
    "anthropic": _ANTHROPIC, "mock": _ANTHROPIC,
    "openai": _OPENAI, "fireworks": _OPENAI, "openai-compatible": _OPENAI,
    "xai": _XAI, "deepseek": _DEEPSEEK, "gemini": _GEMINI,
}


def read_ending(provider: str, stop_reason: str, *, tool_call: bool = False,
                empty: bool = False) -> str:
    """How a remote reply ended, in the words local items use.

    An empty reply is `empty` whatever the provider said (its
    `metadata.empty` has the cause); a reply that ended naturally
    holding a tool call is `tool_call`, since Gemini and the Responses
    API report such a reply as a plain stop."""
    if empty:
        return "empty"
    table = _BY_PROVIDER.get(provider, _OPENAI)
    ended = table.get(str(stop_reason or "").lower(), "other")
    if ended == "end" and tool_call:
        return "tool_call"
    return ended
