from __future__ import annotations

from typing import Any

from mechbench_compute.weights.constants import MODEL_SCOPE


def read_parameters(lm: Any) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    out: dict[str, Any] = {}
    for name, arr in tree_flatten(lm.parameters()):
        out[name.removeprefix(MODEL_SCOPE)] = arr
    return out
