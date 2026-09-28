from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.blocks.set_field import set_field


def rank_guest_records(items: list[dict[str, Any]], rank: str | None,
                       order_by: Sequence[str] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if rank:
        return [set_field(r, rank, i + 1) for i, r in enumerate(items)], {"order_by": [rank]}
    if order_by and all(_holds(r, f) for r in items for f in order_by):
        return items, {"order_by": list(order_by)}
    return items, {}


def _holds(record: dict[str, Any], path: str) -> bool:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True
