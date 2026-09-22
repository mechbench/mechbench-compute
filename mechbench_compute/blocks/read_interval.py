from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_interval(params: Mapping[str, Any]) -> tuple[float, int, int] | None:
    """(level, resamples, seed) when the node asks for an interval."""
    level = params.get("interval")
    if level is None:
        return None
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"interval must be between 0 and 1 exclusive, not {level}")
    return level, int(params.get("resamples", 2000)), int(params.get("seed", 0))
