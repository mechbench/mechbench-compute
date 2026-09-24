from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.directions.is_direction import is_direction


def coerce_array(d: Mapping[str, Any]) -> np.ndarray:
    from mechbench_compute.lexicon import kinds as K

    if isinstance(d, Mapping) and K.item_kind_of(d) is not None:
        items = list(K.items_of(d))
        if len(items) != 1:
            raise ValueError(
                f"expected one direction, not a collection of {len(items)} — "
                "select the one you mean (records/select, records/rank k: 1)")
        d = items[0]
    if not is_direction(d):
        raise ValueError("expected a direction object (kind 'direction/vector')")
    return np.asarray(d["vector"], dtype=np.float32).reshape(-1)
