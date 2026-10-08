from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def truncate(w: Any, item: Mapping[str, Any], name: str) -> Any:
    from mechbench_compute.arrays import read_f32
    from mechbench_compute.intervene.array_ops import read_array_ops

    rank = item.get("rank")
    if rank is None:
        raise ValueError(f"op 'truncate' on {name} needs a `rank`")
    rank = int(rank)
    if w.ndim != 2:
        raise ValueError(f"{name} is not a matrix; there is nothing to truncate")
    arr = read_f32(w)
    if rank >= min(arr.shape):
        return w
    u, sv, vt = np.linalg.svd(arr, full_matrices=False)
    kept = (u[:, :rank] * sv[:rank]) @ vt[:rank]
    return read_array_ops(w).array(kept)
