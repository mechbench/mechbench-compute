from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_trajectory(x: Any, port: str = "trajectory",
                    coords_ok: bool = False) -> Mapping[str, Any]:
    from mechbench_compute.lexicon import kinds as K

    if isinstance(x, Mapping):
        ik = K.item_kind_of(x)
        if ik == "trajectory/point" or (coords_ok and ik == "activations/coordinate"):
            return x
    raise ValueError(f"port {port!r} is not a collection of trajectory/point"
                     + (" or activations/coordinate" if coords_ok else ""))
