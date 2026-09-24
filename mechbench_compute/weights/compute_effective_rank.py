from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_effective_rank(sv: Sequence[float] | np.ndarray) -> float:
    s = np.asarray(sv, dtype=np.float64)
    total = float(s.sum())
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))
