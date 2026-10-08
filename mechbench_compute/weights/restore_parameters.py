from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.weights.resolve_module import resolve_module


def restore_parameters(lm: Any, handle: Sequence[tuple[str, Any]]) -> None:
    if not handle:
        return
    from mechbench_compute.intervene.array_ops import read_array_ops

    xp = read_array_ops(handle[0][1])
    for name, before in reversed(list(handle)):
        module, attr = resolve_module(lm, name)
        xp.write_parameter(module, attr, before)
    xp.settle([getattr(*resolve_module(lm, n)) for n, _ in handle])
