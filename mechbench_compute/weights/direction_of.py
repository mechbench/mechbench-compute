from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _direction_of(item: Mapping[str, Any], name: str) -> np.ndarray:
    from mechbench_compute import directions as dirs

    d = item.get("direction")
    if d is None:
        raise ValueError(
            f"op 'project_out' on {name} needs a `direction` — the thing to "
            "take out of what this module reads or writes.")
    v = np.asarray(dirs.as_array(d), dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError(f"the direction given for {name} is all zeros")
    return v / n
