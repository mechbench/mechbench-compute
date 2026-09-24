from __future__ import annotations

from typing import Any


def resolve_layers(spec: Any, n_layers: int) -> list[int]:
    if spec in (None, "all"):
        return list(range(n_layers))
    layers = [int(spec)] if isinstance(spec, int) else [int(x) for x in spec]
    bad = [i for i in layers if not 0 <= i < n_layers]
    if bad:
        raise ValueError(
            f"layers {bad} out of range for this model (n_layers={n_layers})"
        )
    return layers
