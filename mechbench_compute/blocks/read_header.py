from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_header(collection: Any) -> dict[str, Any]:
    if not isinstance(collection, Mapping):
        return {}
    return {k: v for k, v in collection.items() if k not in ("items", "rows")}
