from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_empty(item: Any) -> Mapping[str, Any] | None:
    """A remote chat item's `metadata.empty` — `{cause, message}` — or
    None when its reply carried prose or a tool call."""
    meta = item.get("metadata") if isinstance(item, Mapping) else None
    return (meta or {}).get("empty") or None
