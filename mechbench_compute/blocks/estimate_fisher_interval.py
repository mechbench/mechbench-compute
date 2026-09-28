from __future__ import annotations

import math
from statistics import NormalDist


def estimate_fisher_interval(rho: float | None, n: int, level: float) -> tuple[float | None, float | None]:
    if rho is None or n < 4 or abs(rho) >= 1.0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + level / 2)
    se = math.sqrt((1 + rho * rho / 2) / (n - 3))
    centre = math.atanh(rho)
    return math.tanh(centre - z * se), math.tanh(centre + z * se)
