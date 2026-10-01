from __future__ import annotations

from collections.abc import Sequence


def estimate_bootstrap_ratio(num: Sequence[float], den: Sequence[float], level: float,
                             resamples: int, seed: int) -> tuple[float | None, float | None]:
    import numpy as np

    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a paired bootstrap needs as many numerators as denominators, at least one")
    point = float(a.mean() / b.mean()) if b.mean() != 0 else None
    if a.size < 2:
        return point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(int(resamples), a.size))
    tops, bottoms = a[idx].mean(axis=1), b[idx].mean(axis=1)
    ok = bottoms != 0
    if not ok.any():
        return None, None
    ratios = tops[ok] / bottoms[ok]
    lo, hi = np.percentile(ratios, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)
