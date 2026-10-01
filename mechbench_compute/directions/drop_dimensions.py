from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def drop_dimensions(rows: np.ndarray, exclude: Sequence[int], *, center: bool = False) -> np.ndarray:
    out = np.array(rows, dtype=np.float32, copy=True)
    out[..., list(exclude)] = 0.0
    if center:
        kept = np.ones(out.shape[-1], dtype=bool)
        kept[list(exclude)] = False
        out[..., kept] -= out[..., kept].mean(axis=-1, keepdims=True)
    return out
