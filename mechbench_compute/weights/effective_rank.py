from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def effective_rank(sv: Sequence[float] | np.ndarray) -> float:
    """exp(H(p)) over the normalised spectrum: 1 for a single
    direction, r for r equal ones, 0 for a delta that is all zero."""
    s = np.asarray(sv, dtype=np.float64)
    total = float(s.sum())
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))
