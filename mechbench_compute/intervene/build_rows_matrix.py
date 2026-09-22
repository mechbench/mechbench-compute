from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.intervene.read_source_items import read_source_items
from mechbench_compute.intervene.spec_error import SpecError


def build_rows_matrix(source: Mapping[str, Any], layer: int | None,
                      point: str | None = None) -> np.ndarray:
    rows = [r for r in read_source_items(source)
            if (layer is None or S.layer_of(r) == layer)
            and (point is None or S.space_of(r).get("point") == point
                 or "space" not in r)]
    if not rows:
        raise SpecError(f"`source` has no vectors at layer {layer}"
                        + (f", point {point!r}" if point else ""))
    return np.array([r["vector"] for r in rows], dtype=np.float32)
