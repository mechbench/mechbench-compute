from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def drop_field(record: Mapping[str, Any], path: str) -> dict[str, Any]:
    head, _, rest = path.partition(".")
    out = dict(record)
    if not rest:
        out.pop(head, None)
        return out
    inner = out.get(head)
    if isinstance(inner, Mapping):
        out[head] = drop_field(inner, rest)
    return out
