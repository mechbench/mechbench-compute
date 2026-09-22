from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.reduce.monoid import Monoid


def monoid_for(block: str, params: Mapping[str, Any] | None = None) -> Monoid | None:
    from mechbench_compute.reduce import MONOIDS
    cls = MONOIDS.get(block)
    if cls is None:
        # An operation in its own file says it can be computed in chunks
        # by defining MONOID there (docs/OPS_LAYOUT.md).
        from mechbench_compute import ops

        cls = getattr(ops.find(block), "MONOID", None)
    if cls is None:
        return None
    m = cls()
    if hasattr(m, "bind"):
        m.bind(params or {})
    return m
