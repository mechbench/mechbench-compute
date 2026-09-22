"""Minimum spanning trees over a similarity structure, as a variety
measure (task 000430, Benji's method).

The question this answers: how VARIED is a corpus, in a domain that has
cluster-attractors — genre, register, a stock plot. Pairwise distance
alone will not say, because a corpus can be far apart on average while
being three tight clumps, and "average distance" reports the same
number for that as for a genuinely even spread.

A minimum spanning tree makes the structure legible. Its edges are the
cheapest set that still connects everything, so within a clump the
edges are short and between clumps there is one long BRIDGE. Then:

  mean edge      the scale of the spread — small means collapsed
  variance       the CLUMPINESS — high means clusters with bridges
  cv             stdev/mean, the same thing made scale-free
  bridges        edges far above the mean: roughly (clusters - 1)

Variance alone is non-monotone in variety and must never be reported by
itself: a collapsed corpus (every story the same) and an evenly varied
one both give LOW variance, for opposite reasons. It is `mean` that
tells those two apart, which is why this block emits them together and
the table puts them side by side.

Prim's algorithm, not scipy: ties break toward the lower index, so the
tree is a deterministic function of the matrix. A run that reproduces
its numbers should reproduce its tree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _distance_from_similarity(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Cosine similarity in [-1, 1] -> distance in [0, 2]."""
    sim = np.array(matrix, dtype=float)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def center_rows(vectors: np.ndarray) -> np.ndarray:
    """Subtract the corpus mean before measuring distance.

    Transformer representations are anisotropic: they occupy a narrow
    cone around one dominant direction, so raw cosine between any two
    of them is mostly a measure of that shared direction rather than
    of how the two differ. Mean-pooled vectors are worse, because
    averaging over a sequence amplifies the common component.

    Measured on experiment 024's frontier corpus: the raw mean MST edge
    over mean-pooled vectors is 0.0048, and 0.5026 after centering — a
    hundredfold. The uncentered numbers were not measuring the corpus,
    they were measuring the cone, and the corpus RANKINGS they produced
    disagreed with each other across layer and pooling choice while the
    centered ones agreed.
    """
    return vectors - vectors.mean(axis=0, keepdims=True)


def _vectors_to_distance(rows: Sequence[Mapping[str, Any]], *,
                         center: bool = False) -> np.ndarray:
    from mechbench_compute import geometry

    vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
    if center:
        vectors = center_rows(vectors)
    return _distance_from_similarity(geometry.cosine_matrix(vectors))


