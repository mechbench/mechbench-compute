from __future__ import annotations

from typing import Any


def sort_group_key(key: tuple) -> tuple:
    return tuple(_place(x) for x in key)


def _place(x: Any) -> tuple:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return (0, float(x), "")
    return (1, 0.0, str(x))
