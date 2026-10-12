from __future__ import annotations

from typing import Any


def read_items(x: Any) -> list[dict[str, Any]]:
    from mechbench_compute import tensors
    from mechbench_compute.lexicon import kinds as K

    if tensors.is_tensor(x):
        return tensors.read_json_items(x)
    return K.items_of(x)
