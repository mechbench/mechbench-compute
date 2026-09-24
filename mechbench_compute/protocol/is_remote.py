from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REMOTE_BLOCKS = ("text/chat", "eval/judge")


def is_remote(block: str, params: Mapping[str, Any]) -> bool:
    if block not in REMOTE_BLOCKS:
        return False
    model = params.get("model") or params.get("judge") or {}
    if isinstance(model, Mapping):
        return bool(model.get("provider"))
    return bool(getattr(model, "is_endpoint", False))
