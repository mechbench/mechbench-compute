from __future__ import annotations

from typing import Any


def read_numbers(value: Any) -> list[float] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)) and value:
        out: list[float] = []
        for x in value:
            got = read_numbers(x)
            if got is None:
                return None
            out.extend(got)
        return out
    return None
