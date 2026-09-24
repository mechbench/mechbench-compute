from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.weights.resolve_module import resolve_module


def restore_parameters(lm: Any, handle: Sequence[tuple[str, Any]]) -> None:
    import mlx.core as mx

    for name, before in reversed(list(handle)):
        module, attr = resolve_module(lm, name)
        setattr(module, attr, before)
    if handle:
        mx.eval([getattr(*resolve_module(lm, n)) for n, _ in handle])
