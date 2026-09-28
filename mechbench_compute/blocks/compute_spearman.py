from __future__ import annotations

import math
from collections.abc import Sequence

from mechbench_compute.blocks.rank_average import rank_average


def compute_spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = rank_average(xs), rank_average(ys)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    num = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(math.fsum((a - mx) ** 2 for a in rx) * math.fsum((b - my) ** 2 for b in ry))
    return num / den if den else None
