from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def take_one(d: Mapping[str, Any]) -> Mapping[str, Any]:
    from mechbench_compute.lexicon import kinds as K

    if isinstance(d, Mapping) and K.item_kind_of(d) is not None:
        items = list(K.items_of(d))
        if len(items) != 1:
            raise ValueError(
                f"expected one direction, not a collection of {len(items)} — "
                "select the one you mean (records/filter, or records/sort with limit: 1)")
        return items[0]
    return d
