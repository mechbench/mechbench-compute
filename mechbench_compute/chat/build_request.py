from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.providers import messages as pm


def build_request(rec: Mapping[str, Any], params: Mapping[str, Any], *,
                  model: str, provider_options: Mapping[str, Any],
                  seed: int | None = None,
                  tools: Sequence[Any] | None = None) -> pm.ChatRequest:
    """One record's request. A record carries either a full `messages`
    conversation or the `system`/`user` fields a corpus record has —
    the same fields `generate` reads, so a protocol can swap a local
    generate node for a chat node without rewriting its corpus. A
    record that carries them under other names goes through
    records/rename first."""
    convo = rec.get("messages")
    if convo is None and "user" in rec:
        convo = [{"role": "user", "content": rec["user"]}]
    if convo is None:
        convo = params.get("messages") or []
    system = str(rec.get("system") or params.get("system") or "")
    return pm.request({
        "model": model,
        "system": system,
        "messages": convo,
        "max_tokens": int(params.get("max_tokens", 1024)),
        "tools": tuple(tools) if tools is not None else (),
        "tool_choice": params.get("tool_choice"),
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "stop": tuple(params.get("stop") or ()),
        "seed": seed,
        "json_mode": bool(params.get("json_mode", False)),
        "logprobs": params.get("logprobs"),
        "provider_options": dict(provider_options),
    })
