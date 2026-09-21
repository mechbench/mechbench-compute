from __future__ import annotations

from typing import Any

from mechbench_compute.weights.constants import _SCOPE


def parameter_names(lm: Any) -> dict[str, Any]:
    """`{name: tensor}` for every parameter of a loaded text decoder,
    named as the module tree names it (`layers.12.self_attn.q_proj.
    weight`), the `model.` scope stripped."""
    from mlx.utils import tree_flatten

    out: dict[str, Any] = {}
    for name, arr in tree_flatten(lm.parameters()):
        out[name.removeprefix(_SCOPE)] = arr
    return out
