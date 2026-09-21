from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.weights.constants import RESIDUAL_SIDE
from mechbench_compute.weights.coords_of import _coords_of
from mechbench_compute.weights.direction_of import _direction_of


def _project_out(w: Any, item: Mapping[str, Any], name: str,
                 strength: float) -> Any:
    """Take a direction out of the side of this weight that faces the
    residual stream: what the module writes (left) or what it reads
    (right). `strength` 1.0 removes it entirely; 0.5 halves it."""
    import mlx.core as mx

    v = _direction_of(item, name)
    coords = _coords_of(name)
    known = RESIDUAL_SIDE.get(str(coords.get("projection")))
    side = str(item.get("side") or (known[0] if known else ""))
    if side not in ("in", "out"):
        raise ValueError(
            f"which side of {name} is the direction in? Its residual side is "
            f"not known, so say `side: \"in\"` (what it reads) or "
            f"`side: \"out\"` (what it writes).")
    dim = w.shape[0] if side == "out" else w.shape[1]
    if len(v) != dim:
        raise ValueError(
            f"the direction is {len(v)} wide and {name}'s {side} side is "
            f"{dim}: a direction only removes from the space it lives in.")
    u = mx.array(v)[:, None] if side == "out" else mx.array(v)[None, :]
    # W − s·(uuᵀ)W on the output side, W − s·W(vvᵀ) on the input side.
    return w - float(strength) * ((u @ (u.T @ w)) if side == "out"
                                  else ((w @ u.T) @ u))
