from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.providers import Budget


def summarize_spend(calls: Sequence[Mapping[str, Any]], budget: Budget, *,
                    provider: str, dry_run: bool, replayed: int) -> dict[str, Any]:
    usage: dict[str, int] = {}
    versions: dict[str, int] = {}
    for c in calls:
        for k, v in (c.get("usage") or {}).items():
            usage[k] = usage.get(k, 0) + int(v)
        mv = str(c.get("model_version") or "")
        versions[mv] = versions.get(mv, 0) + 1
    return {
        "provider": provider,
        "calls": len(calls),
        "cost_usd": round(sum(float(c.get("cost_usd", 0.0)) for c in calls), 8),
        "unpriced_calls": sum(1 for c in calls if c.get("priced") is False),
        "usage": usage,
        "model_versions": versions,
        "budget": budget.to_wire(),
        "throttled_seconds": round(
            sum(float(c.get("throttled_seconds", 0.0)) for c in calls), 3),
        "dry_run": bool(dry_run),
        "replayed": replayed,
    }
