from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.weights.constants import RESIDUAL_SIDE
from mechbench_compute.weights.parse_parameter_coords import parse_parameter_coords
from mechbench_compute.weights.read_direction import read_direction


def project_out(w: Any, item: Mapping[str, Any], name: str,
                strength: float) -> Any:
    from mechbench_compute.intervene.array_ops import read_array_ops

    xp = read_array_ops(w)
    v = read_direction(item, name)
    coords = parse_parameter_coords(name)
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
    u = xp.array(v)[:, None] if side == "out" else xp.array(v)[None, :]
    return w - float(strength) * ((u @ (u.T @ w)) if side == "out"
                                  else ((w @ u.T) @ u))
