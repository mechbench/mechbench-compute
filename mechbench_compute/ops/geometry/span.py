from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="geometry/span",
    summary=(
        "Measure how varied a set of items is with a minimum spanning tree "
        "over their pairwise distances — the spread, its clumpiness, and how "
        "many clusters the bridges imply — for anything a metric compares."
    ),
    description="""\
Average pairwise distance cannot tell three tight clumps from an even
spread; a minimum spanning tree can. Its edges are the cheapest set that
still connects every point, so within a clump edges are short and between
clumps there is one long **bridge**. Per group the block reports the mean
edge (the scale of the spread — small means collapsed), the variance and
coefficient of variation (clumpiness), and the number of bridges — edges
more than `bridge_sigma` standard deviations above the mean — which is
roughly the cluster count minus one. Read `mean` and `variance` together:
a collapsed corpus and an evenly varied one both have low variance, for
opposite reasons.

The tree is built on a `geometry/similarity` collection, so it stands over
whatever that op compared: a corpus of vectors by centred cosine, the
adapters' axes, a grid of decision reads by Jensen–Shannon, the design's
records by how many factors differ. A distance metric is used as it is; a
cosine as `1 − cosine`; any other similarity as `max − value`. A metric
that is not symmetric (`kl`) is refused by name. The tree is built
deterministically (ties break toward the lower index), so a run that
reproduces its numbers reproduces its tree.
""",
    inputs=(
        In("similarity", "geometry/similarity",
           "The pairwise matrices, one per group, from `geometry/compare`.",
           many=True),
    ),
    output=Output('geometry/mst', collection=True, doc='One item per group: `group`, `layer`/`head` when the group is a space, `n`, `n_edges`, `mean`, `variance`, `stdev`, `cv`, `total`, `min`, `max`, `bridge_threshold`, `bridges`, `components_after_cut`, `ids`, `labels`, and `edges` as `[i, j, weight]` when kept. The header carries `metric`, `options`, `over` (the kind compared), `bridge_sigma` and `axis`. `records/table` reads the items as its rows.'),
    params=(
        P("bridge_sigma", "float",
          "How many standard deviations above the mean edge an edge must be "
          "to count as a bridge between clusters.",
          2.0),
        P("keep_edges", "bool",
          "Include every tree edge (`[i, j, weight]`) in the output, not "
          "only the statistics.",
          True),
    ),
    example={"bridge_sigma": 2.0},
    example_inputs={"similarity": {"$ref": {"bench": "you/lab/similarity"}}},
)


def run(ctx, inputs, params):
    return build_span_trees(inputs, params)


#: How far above the mean an edge must sit to count as a bridge between
#: clusters rather than a step within one.
DEFAULT_BRIDGE_SIGMA = 2.0


def grow_minimum_spanning_tree(distance: np.ndarray) -> list[tuple[int, int, float]]:
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


def measure_tree(edges: Sequence[tuple[int, int, float]], *,
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


def _read_distance_matrix(entry: Mapping[str, Any], header: Mapping[str, Any]) -> np.ndarray:
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


def build_span_trees(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`geometry/span`: a tree per group of a `geometry/similarity`
    collection, whatever metric produced it — the metric and its
    options ride along from the similarity's header."""
    bridge_sigma = float(params.get("bridge_sigma", DEFAULT_BRIDGE_SIGMA))
    keep_edges = bool(params.get("keep_edges", True))
    from mechbench_compute.lexicon import kinds as K

    src = inputs.get("similarity")
    if not (isinstance(src, Mapping) and K.item_kind_of(src) == "geometry/similarity"):
        raise ValueError(
            "geometry/span needs a collection of geometry/similarity on its "
            f"`similarity` port — got {type(src).__name__}")
    if src.get("symmetric") is False:
        raise ValueError(
            f"geometry/span needs a symmetric metric; {src.get('metric')!r} is not "
            "(m(a, b) ≠ m(b, a)) — compare by a symmetric one, such as "
            "jensen-shannon")

    out_groups = []
    for entry in K.items_of(src):
        distance = _read_distance_matrix(entry, src)
        edges = grow_minimum_spanning_tree(distance)
        stats = measure_tree(edges, bridge_sigma=bridge_sigma)
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
    # One item per group; `records/tabulate` reads the items directly.
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
