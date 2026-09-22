"""The run's bill: total, per provider, per node."""

from __future__ import annotations

from typing import Any


def _spend_total(by_node: dict[str, Any]) -> dict[str, Any]:
    """The run's bill: total, per provider, per node (task 000337)."""
    by_provider: dict[str, dict[str, Any]] = {}
    total = 0.0
    calls = 0
    for s in by_node.values():
        p = str(s.get("provider", ""))
        slot = by_provider.setdefault(p, {"cost_usd": 0.0, "calls": 0})
        slot["cost_usd"] = round(slot["cost_usd"] + float(s.get("cost_usd", 0.0)), 8)
        slot["calls"] += int(s.get("calls", 0))
        total += float(s.get("cost_usd", 0.0))
        calls += int(s.get("calls", 0))
    return {"cost_usd": round(total, 8), "calls": calls,
            "by_provider": by_provider,
            "by_node": {k: {"cost_usd": round(float(v.get("cost_usd", 0.0)), 8),
                            "calls": int(v.get("calls", 0)),
                            "provider": v.get("provider", "")}
                        for k, v in by_node.items()},
            "dry_run": all(bool(v.get("dry_run")) for v in by_node.values())}
