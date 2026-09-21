from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.weights.module_of import _module_of


def restore_parameters(lm: Any, handle: Sequence[tuple[str, Any]]) -> None:
    """Undo `edit_parameters` by reinstalling the tensors it kept."""
    import mlx.core as mx

    for name, before in reversed(list(handle)):
        module, attr = _module_of(lm, name)
        setattr(module, attr, before)
    if handle:
        mx.eval([getattr(*_module_of(lm, n)) for n, _ in handle])
