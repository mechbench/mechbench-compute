from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import shapes as S


def _models_of(vectors: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The distinct models the rows were captured from, in order."""
    seen: dict[str, None] = {}
    for r in rows:
        m = S.space_of(r, header=vectors).get("model")
        if m is not None:
            seen[str(m)] = None
    return list(seen)
