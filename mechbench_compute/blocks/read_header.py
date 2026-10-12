from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_header(collection: Any) -> dict[str, Any]:
    from mechbench_compute import tensors

    if not isinstance(collection, Mapping):
        return {}
    header = tensors.strip_header(collection) if tensors.is_tensor(collection) else collection
    return {k: v for k, v in header.items() if k not in ("items", "rows")}
