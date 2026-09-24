from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.weights.resolve_module import resolve_module
from mechbench_compute.weights.read_parameters import read_parameters
from mechbench_compute.weights.project_out import project_out
from mechbench_compute.weights.select_points import select_points
from mechbench_compute.weights.truncate import truncate

WEIGHT_OPS: tuple[str, ...] = ("zero", "scale", "project_out", "truncate")


def edit_parameters(lm: Any, items: Sequence[Mapping[str, Any]],
                    factor: float = 1.0) -> list[tuple[str, Any]]:
    import mlx.core as mx

    tensors = read_parameters(lm)
    handle: list[tuple[str, Any]] = []
    for item in items:
        point = str(item.get("parameter") or item.get("point") or "")
        names = select_points(tensors, [point])
        op = str(item.get("op", "zero"))
        if op not in WEIGHT_OPS:
            raise ValueError(
                f"unknown weight op {op!r}; one of {', '.join(WEIGHT_OPS)}. "
                f"(The activation ops act during a forward pass; a parameter "
                f"edit lasts for the node.)")
        strength = float(item.get("strength", 1.0)) * float(factor)
        for name in names:
            module, attr = resolve_module(lm, name)
            before = getattr(module, attr)
            handle.append((name, before))
            w = before.astype(mx.float32)
            if op == "zero":
                after = mx.zeros_like(w)
            elif op == "scale":
                after = w * strength
            elif op == "project_out":
                after = project_out(w, item, name, strength)
            else:
                after = truncate(w, item, name)
            setattr(module, attr, after.astype(before.dtype))
    mx.eval([getattr(*resolve_module(lm, n)) for n, _ in handle])
    return handle
