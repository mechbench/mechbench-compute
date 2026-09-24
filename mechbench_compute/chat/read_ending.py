from __future__ import annotations

_ANTHROPIC = {
    "end_turn": "end", "stop_sequence": "stop", "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens", "tool_use": "tool_call",
    "refusal": "filtered",
}
_OPENAI = {
    "stop": "end", "length": "max_tokens", "tool_calls": "tool_call",
    "function_call": "tool_call", "content_filter": "filtered",
    "completed": "end", "max_output_tokens": "max_tokens",
}
_XAI = {**_OPENAI, "end_turn": "end"}
_DEEPSEEK = {
    "stop": "end", "length": "max_tokens", "tool_calls": "tool_call",
    "content_filter": "filtered",
}
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
    if empty:
        return "empty"
    table = _BY_PROVIDER.get(provider, _OPENAI)
    ended = table.get(str(stop_reason or "").lower(), "other")
    if ended == "end" and tool_call:
        return "tool_call"
    return ended
