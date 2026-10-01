from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.directions.is_direction import is_direction
from mechbench_compute.directions.take_one import take_one


def coerce_array(d: Mapping[str, Any]) -> np.ndarray:
    d = take_one(d)
    if not is_direction(d):
        raise ValueError("expected a direction object (kind 'direction/vector')")
    return np.asarray(d["vector"], dtype=np.float32).reshape(-1)
