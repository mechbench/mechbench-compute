from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import shapes as S


def select_layer_items(vectors: Mapping[str, Any], layer: int) -> list[Mapping[str, Any]]:
    from mechbench_compute.lexicon import kinds as K

    if not isinstance(vectors, Mapping) or K.item_kind_of(vectors) != "activations/vector":
        raise ValueError("expected a collection of activations/vector")
    rows = [r for r in K.items_of(vectors) if S.layer_of(r) == layer]
    if not rows:
        raise ValueError(f"the vectors collection has no items at layer {layer}")
    return rows
