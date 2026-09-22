from __future__ import annotations

from typing import Any


def build_collection(items: list[dict[str, Any]], **header: Any) -> dict[str, Any]:
    """A `collection` of `records/record`."""
    from mechbench_compute.lexicon import kinds as K

    return K.collection("records/record", items, **header)
