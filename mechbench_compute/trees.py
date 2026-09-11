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


def mst(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`~canonical/ops/vectors/mst/1`.

    Takes either a `similarity_matrix` (the output of
    `vectors/similarity`) on the `matrix` port, or a
    `residual_vectors` record on the `vectors` port — so the block
    composes with or without the intermediate similarity node.
    """
    bridge_sigma = float(params.get("bridge_sigma", DEFAULT_BRIDGE_SIGMA))
    keep_edges = bool(params.get("keep_edges", True))
    # Off by default so stored results keep their numbers; every new
    # protocol should turn it on. See `center_rows`.
    center = bool(params.get("center", False))
    src = (inputs.get("matrix") or inputs.get("similarity")
           or params.get("matrix"))
    vectors = inputs.get("vectors") or params.get("vectors")

    groups: list[dict[str, Any]] = []
    if center and isinstance(src, Mapping) and src.get("kind") == "similarity_matrix":
        raise ValueError(
            "vectors/mst cannot center a similarity_matrix — the vectors "
            "are already gone. Feed it `vectors` instead, or center "
            "upstream in vectors/similarity.")
    if isinstance(src, Mapping) and src.get("kind") == "similarity_matrix":
        for entry in src.get("layers", []):
            groups.append({
                "layer": entry.get("layer"),
                **({"head": entry["head"]} if entry.get("head") is not None else {}),
                "ids": entry.get("ids", []),
                "labels": entry.get("labels", []),
                "distance": _distance_from_similarity(entry["matrix"]),
            })
    elif isinstance(vectors, Mapping) and vectors.get("kind") == "residual_vectors":
        rows = vectors.get("rows") or []
        keys = sorted({(r.get("layer"), r.get("head")) for r in rows},
                      key=lambda t: (t[0] if t[0] is not None else -1,
                                     t[1] if t[1] is not None else -1))
        for layer, head in keys:
            sub = [r for r in rows
                   if r.get("layer") == layer and r.get("head") == head]
            if len(sub) < 2:
                continue
            groups.append({
                "layer": layer,
                **({"head": head} if head is not None else {}),
                "ids": [r.get("id") for r in sub],
                "labels": [r.get("label") for r in sub],
                "distance": _vectors_to_distance(sub, center=center),
            })
    else:
        raise ValueError(
            "vectors/mst needs a `similarity_matrix` on its `matrix` port "
            "or a `residual_vectors` record on `vectors` — got "
            f"{type(src or vectors).__name__}")

    out_layers, rows = [], []
    for g in groups:
        edges = minimum_spanning_tree(g["distance"])
        stats = tree_stats(edges, bridge_sigma=bridge_sigma)
        entry: dict[str, Any] = {
            "layer": g["layer"], "n": len(g["ids"]), **stats,
            "ids": g["ids"], "labels": g["labels"],
        }
        if "head" in g:
            entry["head"] = g["head"]
        if keep_edges:
            entry["edges"] = [[i, j, round(w, 6)] for i, j, w in edges]
        out_layers.append(entry)
        rows.append({k: v for k, v in entry.items()
                     if k not in ("ids", "labels", "edges")})

    return {
        "kind": "mst_summary",
        "name": params.get("name", "mst"),
        "metric": "centered_cosine_distance" if center else "cosine_distance",
        "centered": center,
        "bridge_sigma": bridge_sigma,
        "layers": out_layers,
        # The per-layer statistics again as a flat record list, for
        # `table/from-records` and anything else that wants rows rather
        # than the nested `layers`. This is NOT a `metric_table` (that
        # kind needs `columns` and `row_axis`) and it is not what the
        # UI renders: `mst_summary` has its own view, which draws the
        # edge-weight histogram the statistics summarize.
        "rows": rows,
        "description": (
            "Minimum spanning tree over pairwise cosine distance. `mean` is "
            "the scale of the spread and `variance` its clumpiness; they are "
            "read together, because a collapsed corpus and an evenly varied "
            "one both have low variance for opposite reasons."
        ),
    }


PURE_TREE_BLOCKS = {"~canonical/ops/vectors/mst/1": mst}
