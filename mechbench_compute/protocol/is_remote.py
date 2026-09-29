from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_remote(block: Any, params: Mapping[str, Any]) -> bool:
    if not hasattr(block, "op"):
        from mechbench_compute.registry import REGISTRY

        block = REGISTRY.find(block)
    if block is None or "provider.chat" not in block.op.needs:
        return False
    model = params.get("model") or params.get("judge") or {}
    if isinstance(model, Mapping):
        return bool(model.get("provider"))
    return bool(getattr(model, "is_endpoint", False))
