from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def encode_features(x: np.ndarray, weights: Mapping[str, Any]) -> np.ndarray:
    pre = x @ weights["w_enc"] + weights["b_enc"]
    fn = weights["activation"]["fn"]
    if fn == "jumprelu":
        return np.where(pre > weights["threshold"], pre, np.float32(0)).astype(np.float32)
    f = np.maximum(pre, np.float32(0))
    if fn == "topk":
        k = int(weights["activation"]["k"])
        if k < f.shape[1]:
            cut = np.argsort(-f, axis=1, kind="stable")[:, k:]
            np.put_along_axis(f, cut, np.float32(0), axis=1)
    return f.astype(np.float32)
