from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def set_field(record: Mapping[str, Any], path: str, value: Any) -> dict[str, Any]:
    head, _, rest = path.partition(".")
    out = dict(record)
    if not rest:
        out[head] = value
        return out
    inner = out.get(head)
    out[head] = set_field(inner if isinstance(inner, Mapping) else {}, rest, value)
    return out
