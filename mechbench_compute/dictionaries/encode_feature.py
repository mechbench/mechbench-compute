from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.dictionaries.encode_features import encode_features


def encode_feature(x: np.ndarray, feature: Mapping[str, Any]) -> np.ndarray:
    if feature["weights"] is not None:
        return encode_features(x, feature["weights"])[:, feature["index"]]
    pre = x @ feature["encoder"] + feature["b_enc"]
    if feature["activation"]["fn"] == "jumprelu":
        return np.where(pre > feature["threshold"], pre, np.float32(0)).astype(np.float32)
    return np.maximum(pre, np.float32(0)).astype(np.float32)
