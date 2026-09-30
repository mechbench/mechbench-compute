from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SHAPE_KEYS = ("n", "b", "k", "t", "variant")


def write_shape_key(shape: Mapping[str, Any]) -> str:
    unknown = sorted(set(shape) - set(SHAPE_KEYS))
    if unknown:
        raise ValueError(f"a calibration shape has n, b, k, t and variant, not {unknown}")
    return ",".join(f"{k}={shape[k]}" for k in SHAPE_KEYS if shape.get(k) is not None)
