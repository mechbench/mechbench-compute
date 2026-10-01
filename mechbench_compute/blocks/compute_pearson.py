from __future__ import annotations

import math
from collections.abc import Sequence


def compute_pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    num = math.fsum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    den = math.sqrt(math.fsum((a - mx) ** 2 for a in xs) * math.fsum((b - my) ** 2 for b in ys))
    return max(-1.0, min(1.0, num / den)) if den else None
