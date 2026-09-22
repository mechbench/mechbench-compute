"""Whether a node's work happens on somebody else's machine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The blocks whose work is somebody else's network, not this machine's.
#: Two of these that do not depend on each other have no reason to wait
#: for each other: the time is latency. Everything else stays serial — a
#: local model node MUST (one model in memory, one fused adapter at a
#: time), and a pure block takes microseconds, where a thread would be
#: risk for no gain.
REMOTE_BLOCKS = ("text/chat", "eval/judge")


def is_remote(block: str, params: Mapping[str, Any]) -> bool:
    """Whether this node's work happens on somebody else's machine: a
    chat-shaped block whose model reference names a provider."""
    if block not in REMOTE_BLOCKS:
        return False
    model = params.get("model") or params.get("judge") or {}
    if isinstance(model, Mapping):
        return bool(model.get("provider"))
    return bool(getattr(model, "is_endpoint", False))
