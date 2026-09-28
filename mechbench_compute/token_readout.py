from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute.directions.coerce_array import coerce_array


class TokenReadout:
    def __init__(self, model: Any, direction: Mapping[str, Any]) -> None:
        from mechbench_compute.lexicon import kinds as K

        if K.item_kind_of(direction) is not None:
            items = list(K.items_of(direction))
            if len(items) != 1:
                raise ValueError(f"project: expected one direction, not a collection of {len(items)}")
            direction = items[0]
        sp = S.space_of(direction)
        if sp.get("layer") is None:
            raise ValueError("project: the direction names no layer, so there is no residual to read it at")
        self.point = P.residual(sp.get("point"))
        self.layer = int(sp["layer"])
        self.hook = f"blocks.{self.layer}.{self.point}"
        self.vector = coerce_array(direction).astype(np.float32)
        d_model = getattr(getattr(model, "arch", None), "d_model", None)
        if d_model is not None and self.vector.shape[0] != int(d_model):
            raise ValueError(
                f"project: the direction has {self.vector.shape[0]} dims; the model has {d_model}")
        self.direction = direction
        self.coords: list[float] = []

    def read(self, cache: Mapping[str, Any]) -> float:
        row = cache[self.hook][0, -1]
        if not isinstance(row, np.ndarray):
            import mlx.core as mx

            row = row.astype(mx.float32)
        row = np.asarray(row, dtype=np.float32)
        coord = round(float(row @ self.vector), 6)
        self.coords.append(coord)
        return coord

    def record(self, pieces: list[str]) -> dict[str, Any]:
        return {"direction": S.direction_ref(self.direction),
                "tokens": list(pieces), "coords": list(self.coords[: len(pieces)])}


def streaming(on_token: Any, readout: TokenReadout | None, pieces: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if on_token:
        out["on_token"] = on_token
    if readout is not None:
        out["readout"] = readout
        out["pieces_out"] = pieces
    return out
