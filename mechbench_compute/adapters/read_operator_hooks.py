from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mechbench_compute.adapters.attach_operators import SLOT


def _build_hook(layer: Any, after: Callable | None) -> Callable:
    def operate(act: Any, info: Any) -> Any:
        for module in layer[SLOT]:
            act = module(act)
        if after is None:
            return act
        changed = after(act, info)
        return act if changed is None else changed

    return operate


def read_operator_hooks(lm, hooks: dict[str, Callable]) -> dict[str, Callable]:
    layers = getattr(getattr(lm, "model", None), "layers", None) or ()
    out = hooks
    for i, layer in enumerate(layers):
        if isinstance(layer, dict) and SLOT in layer:
            name = f"blocks.{i}.resid_post"
            out = {**out, name: _build_hook(layer, out.get(name))}
    return out
