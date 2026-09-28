from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any


def estimate_paired_difference(
    a_items: Sequence[tuple[Hashable, float]],
    b_items: Sequence[tuple[Hashable, float]],
    paired: bool,
    rng: Any,
    resamples: int,
    level: float,
) -> dict[str, float] | None:
    import numpy as np

    if paired:
        a_by = dict(a_items)
        b_by = dict(b_items)
        keys = sorted((k for k in a_by if k in b_by), key=str)
        if not keys:
            return None
        a = np.array([a_by[k] for k in keys])
        b = np.array([b_by[k] for k in keys])
        n = len(keys)
        diffs = a - b
        point = float(diffs.mean())
        if n >= 2:
            idx = rng.integers(0, n, size=(resamples, n))
            boots = diffs[idx].mean(axis=1)
        else:
            boots = np.full(resamples, point)
    else:
        a = np.sort([v for _, v in a_items])
        b = np.sort([v for _, v in b_items])
        n = min(len(a), len(b))
        point = float(a.mean() - b.mean())
        if len(a) >= 2 and len(b) >= 2:
            ia = rng.integers(0, len(a), size=(resamples, len(a)))
            ib = rng.integers(0, len(b), size=(resamples, len(b)))
            boots = a[ia].mean(axis=1) - b[ib].mean(axis=1)
        else:
            boots = np.full(resamples, point)
    lo, hi = np.percentile(boots, [50 * (1 - level), 50 * (1 + level)])
    return {
        "n": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": point,
        "lo": float(lo),
        "hi": float(hi),
        "share_positive": float((boots > 0).mean()),
    }
