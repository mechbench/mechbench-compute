from __future__ import annotations

from typing import Any


def describe_spend_cap(err: Any, *, op: str, node: str | None, at: str) -> str:
    where = f" ({err.provider}/{err.model})" if getattr(err, "provider", "") else ""
    named = f"{op} {node!r}" if node else op
    return (f"{named} stopped at {at}: budget cap ${err.cap_usd:.4f} would be exceeded{where}: "
            f"${err.spent_usd:.4f} already spent across the body's calls and this call's worst case is "
            f"${err.estimate_usd:.4f}. Raise budget_usd on the {op.split('/')[-1]}, or lower the body's "
            f"max_tokens.")
