from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _distance_from_similarity(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    sim = np.array(matrix, dtype=float)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def center_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors - vectors.mean(axis=0, keepdims=True)


def _vectors_to_distance(rows: Sequence[Mapping[str, Any]], *,
                         center: bool = False) -> np.ndarray:
    from mechbench_compute import geometry

    vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
    if center:
        vectors = center_rows(vectors)
    return _distance_from_similarity(geometry.cosine_matrix(vectors))
