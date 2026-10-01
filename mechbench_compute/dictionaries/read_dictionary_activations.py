from __future__ import annotations

from typing import Any

import numpy as np

from mechbench_compute import points
from mechbench_compute._mlx import mx


def read_dictionary_activations(model: Any, ids: Any, point: str, layer: int) -> np.ndarray:
    declared = tuple(model.architecture.layer_points)
    if point == "attn.o_in" and point not in declared and "attn.per_head_out" in declared:
        name = f"blocks.{layer}.attn.per_head_out"
        t = model.run(ids, capture=[name]).cache[name].astype(mx.float32)
        heads = np.array(t)[0]
        return heads.transpose(1, 0, 2).reshape(heads.shape[1], -1)
    if point not in declared or points.LAYOUT[point][1] is not None:
        raise ValueError(f"the dictionary reads {point}, which {model.architecture.name} does not offer "
                         f"at its support level; its points are {', '.join(declared)}")
    name = f"blocks.{layer}.{point}"
    t = model.run(ids, capture=[name]).cache[name].astype(mx.float32)
    return np.array(t)[0]
