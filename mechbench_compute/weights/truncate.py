from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def truncate(w: Any, item: Mapping[str, Any], name: str) -> Any:
    import mlx.core as mx

    rank = item.get("rank")
    if rank is None:
        raise ValueError(f"op 'truncate' on {name} needs a `rank`")
    rank = int(rank)
    if w.ndim != 2:
        raise ValueError(f"{name} is not a matrix; there is nothing to truncate")
    arr = np.array(w, dtype=np.float32)
    if rank >= min(arr.shape):
        return w
    u, sv, vt = np.linalg.svd(arr, full_matrices=False)
    kept = (u[:, :rank] * sv[:rank]) @ vt[:rank]
    return mx.array(kept)
