from __future__ import annotations

from collections.abc import Mapping
from typing import Any

KIND = "adapter/operator"


def is_operator(payload: Any) -> bool:
    return isinstance(payload, Mapping) and payload.get("kind") == KIND
