from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.directions.as_array import as_array
from mechbench_compute.directions.make import make
from mechbench_compute.directions.same_space import same_space
from mechbench_compute.directions.space_of import space_of


def add(directions: Sequence[Mapping[str, Any]],
        weights: Sequence[float] | None = None) -> dict[str, Any]:
    """Weighted sum of directions in one space, re-normalized.

    The composition primitive: steering along "formal" and "terse" at
    once is their sum, and `weights` sets the mix. Every input must share
    a space — adding across spaces is meaningless, and `same_space`
    refuses it rather than returning a plausible vector.
    """
    if not directions:
        raise ValueError("add needs at least one direction")
    ws = [1.0] * len(directions) if weights is None else [float(w) for w in weights]
    if len(ws) != len(directions):
        raise ValueError("weights must match directions")
    for d in directions[1:]:
        same_space(directions[0], d)
    v = sum(w * as_array(d) for w, d in zip(ws, directions, strict=True))
    return make(v, space_of(directions[0]), method="add",
                sources=[str(d.get("derivation", {}).get("method")) for d in directions],
                extra={"weights": ws})
