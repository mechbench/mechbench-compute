from __future__ import annotations

import math
from statistics import NormalDist


def estimate_wilson(k: int, n: int, level: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + level / 2)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)
