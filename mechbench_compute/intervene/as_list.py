from __future__ import annotations

from typing import Any


def _as_list(x: Any) -> list[int]:
    if isinstance(x, int):
        return [x]
    return [int(i) for i in x]
