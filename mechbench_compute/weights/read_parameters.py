from __future__ import annotations

from typing import Any

from mechbench_compute.weights.constants import MODEL_SCOPE


def read_parameters(lm: Any) -> dict[str, Any]:
    if hasattr(getattr(lm, "model", lm), "named_parameters"):
        from mechbench_compute.torch_backend.parameters import (
            read_parameters as read_torch,
        )

        return read_torch(lm)
    from mlx.utils import tree_flatten

    out: dict[str, Any] = {}
    for name, arr in tree_flatten(lm.parameters()):
        out[name.removeprefix(MODEL_SCOPE)] = arr
    return out
