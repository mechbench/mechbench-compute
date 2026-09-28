from __future__ import annotations

from collections.abc import Sequence


def estimate_bootstrap_mean(values: Sequence[float], level: float, resamples: int,
                    seed: int) -> tuple[float, float]:
    import numpy as np

    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.size < 2:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(int(resamples), v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)
