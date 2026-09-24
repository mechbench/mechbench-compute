from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_empty(item: Any) -> Mapping[str, Any] | None:
    meta = item.get("metadata") if isinstance(item, Mapping) else None
    return (meta or {}).get("empty") or None
