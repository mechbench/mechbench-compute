from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.providers import messages as pm


def build_request(rec: Mapping[str, Any], params: Mapping[str, Any], *,
                  model: str, provider_options: Mapping[str, Any],
                  seed: int | None = None,
                  tools: Sequence[Any] | None = None,
                  api: str | None = None) -> pm.ChatRequest:
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
        "api": api,
        "effort": params.get("effort"),
        "reasoning_display": params.get("reasoning_display"),
        "prompt_cache": params.get("prompt_cache"),
    })
