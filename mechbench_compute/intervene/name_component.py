from __future__ import annotations

from typing import Any


def name_component(point: str, layer: int, head: int | None, position: Any) -> str:
    where = "all" if position in (None, "all") else str(int(position))
    who = "" if head is None else f".H{int(head)}"
    return f"L{int(layer)}.{point}{who}@{where}"
