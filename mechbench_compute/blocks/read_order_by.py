from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_order_by(collection: Any) -> list[str] | None:
    if not isinstance(collection, Mapping):
        return None
    order = collection.get("order_by")
    return [str(f) for f in order] if isinstance(order, list) and order else None
