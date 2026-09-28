from __future__ import annotations

from typing import Any

from mechbench_compute.blocks.set_field import set_field


def rank_guest_records(items: list[dict[str, Any]], rank: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rank:
        return items, {}
    return [set_field(r, rank, i + 1) for i, r in enumerate(items)], {"order_by": [rank]}
