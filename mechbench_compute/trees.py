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

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import shapes as S

#: How far above the mean an edge must sit to count as a bridge between
#: clusters rather than a step within one.
DEFAULT_BRIDGE_SIGMA = 2.0


def minimum_spanning_tree(distance: np.ndarray) -> list[tuple[int, int, float]]:
    """Prim's, deterministic. Returns (i, j, weight) with i < j by
    construction of the frontier, in the order the tree grew."""
    n = int(distance.shape[0])
    if n < 2:
        return []
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = np.array(distance[0], dtype=float)
    parent = np.zeros(n, dtype=int)
    edges: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        masked = np.where(in_tree, np.inf, best)
        # argmin returns the FIRST minimum, so ties go to the lower
        # index and the tree is reproducible.
        j = int(np.argmin(masked))
        if not np.isfinite(masked[j]):
            break                      # disconnected: nothing reachable
        edges.append((int(parent[j]), j, float(best[j])))
        in_tree[j] = True
        closer = (distance[j] < best) & ~in_tree
        parent[closer] = j
        best = np.where(closer, distance[j], best)
    return edges


def tree_stats(edges: Sequence[tuple[int, int, float]], *,
               bridge_sigma: float = DEFAULT_BRIDGE_SIGMA) -> dict[str, Any]:
    """The numbers the measure is about. `mean` and `variance` travel
    together on purpose — see the module docstring."""
    weights = [w for _, _, w in edges]
    if not weights:
        return {"n_edges": 0}
    mean = float(np.mean(weights))
    variance = float(np.var(weights))
    stdev = math.sqrt(variance)
    threshold = mean + bridge_sigma * stdev
    return {
        "n_edges": len(weights),
        "mean": round(mean, 6),
        "variance": round(variance, 6),
        "stdev": round(stdev, 6),
        # Scale-free, so corpora embedded at different layers (whose
        # absolute cosine distances differ) stay comparable.
        "cv": round(stdev / mean, 6) if mean > 0 else 0.0,
        "total": round(float(np.sum(weights)), 6),
        "max": round(float(np.max(weights)), 6),
        "min": round(float(np.min(weights)), 6),
        "bridge_threshold": round(threshold, 6),
        "bridges": int(sum(w > threshold for w in weights)),
        # Cutting the bridges leaves this many components — a cluster
        # count nobody had to choose a k for.
        "components_after_cut": int(sum(w > threshold for w in weights)) + 1,
    }


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


def _distance_of(entry: Mapping[str, Any], header: Mapping[str, Any]) -> np.ndarray:
    """The distance matrix a similarity item stands for: a distance
    metric as it is; cosine as 1 − s (what the tree has always been
    built on); any other similarity as max − s."""
    m = np.array(entry["matrix"], dtype=float)
    kind = header.get("metric_kind") or ("similarity" if header.get("metric") in (None, "cosine") else "distance")
    if kind == "distance":
        dist = m.copy()
    elif header.get("metric", "cosine") == "cosine":
        dist = 1.0 - m
    else:
        dist = float(m.max()) - m
    np.fill_diagonal(dist, 0.0)
    return dist


def mst(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`geometry/mst`: a tree per group of a `geometry/similarity`
    collection, whatever metric produced it — the metric and its
    options ride along from the similarity's header."""
    bridge_sigma = float(params.get("bridge_sigma", DEFAULT_BRIDGE_SIGMA))
    keep_edges = bool(params.get("keep_edges", True))
    from mechbench_compute.lexicon import kinds as K

    src = inputs.get("similarity")
    if not (isinstance(src, Mapping) and K.item_kind_of(src) == "geometry/similarity"):
        raise ValueError(
            "geometry/mst needs a collection of geometry/similarity on its "
            f"`similarity` port — got {type(src).__name__}")
    if src.get("symmetric") is False:
        raise ValueError(
            f"geometry/mst needs a symmetric metric; {src.get('metric')!r} is not "
            "(m(a, b) ≠ m(b, a)) — compare by a symmetric one, such as "
            "jensen-shannon")

    out_groups = []
    for entry in K.items_of(src):
        distance = _distance_of(entry, src)
        edges = minimum_spanning_tree(distance)
        stats = tree_stats(edges, bridge_sigma=bridge_sigma)
        ids = list(entry.get("ids", []))
        item: dict[str, Any] = {"n": len(ids), **stats, "ids": ids,
                                "labels": list(entry.get("labels", []))}
        for k in ("group", "layer", "head"):
            if entry.get(k) is not None:
                item[k] = entry[k]
        if keep_edges:
            item["edges"] = [[i, j, round(w, 6)] for i, j, w in edges]
        out_groups.append(item)

    metric = src.get("metric", "cosine")
    options = dict(src.get("options") or {})
    # One item per group; `records/table` reads the items directly.
    return K.collection(
        "geometry/mst", out_groups,
        name=params.get("name", "mst"),
        metric=metric,
        options=options,
        over=src.get("over"),
        bridge_sigma=bridge_sigma,
        axis=src.get("axis"),
        description=(
            f"Minimum spanning tree over pairwise {metric} distance. `mean` is "
            "the scale of the spread and `variance` its clumpiness; they are "
            "read together, because a collapsed corpus and an evenly varied "
            "one both have low variance for opposite reasons."
        ),
    )


PURE_TREE_BLOCKS = {"geometry/mst": mst}
