from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _pair(record: Mapping[str, Any]) -> tuple[str, str]:
    """A pair's two prompts, `a` and `b` — or the retired `clean` and
    `corrupt`."""
    a = record.get("a", record.get("clean"))
    b = record.get("b", record.get("corrupt"))
    if not (isinstance(a, str) and isinstance(b, str) and a and b):
        raise ValueError(
            f"pair {record.get('id')!r} needs prompt fields `a` and `b`")
    return a, b
