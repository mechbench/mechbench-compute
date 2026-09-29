from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F, SPACE

KIND = Kind(
    "geometry/similarity",
    "The pairwise matrix of a collection's items under one of their kind's metrics — cosine over vectors, Jensen–Shannon over distributions, hamming over records — one item per group, with the metric and its options recorded and, when the items carry a value on the grouping axis, how well the groups separate.",
    fields={"group": F("string", "The group's name: `layer=12`, `layer=3,head=1`, `genre=noir`, or `all`."),
            "layer": F("integer", "The layer, for a group of vectors from one space."),
            "head": F("integer", "The head, for a per-head group."),
            "space": SPACE,
            "ids": F("array", "The item ids, in matrix order.", items={}),
            "labels": F("array", "The items' values on the grouping `axis`, in matrix order.", items={}),
            "matrix": F("array", "The metric's value, `[i][j]`, at full precision.", items={"type": "array"}),
            "pairs": F("array", "Every pair `{a, b, value}`, most alike first, for a group of at most thirty-two items.", items={"type": "object"}),
            "separation": F("object", "`{intra, inter, gap}`: the mean value within groups, between groups, and how much the groups stand apart."),
            "nn_purity": F("number", "Share of items whose nearest neighbour shares their group."),
            "silhouette": F("number", "The silhouette score, when computable.")},
    key=("group",),
    header={"metric": "The metric's name.", "metric_kind": "`similarity` or `distance`: which way larger means.",
            "symmetric": "Whether m(a, b) = m(b, a).", "options": "The metric's options as applied.",
            "over": "The kind of the items compared.", "by": "The header axis the groups were formed on.",
            "axis": "The coordinate the separation reads.",
            "position": "Which position the vectors were read at, when vectors.", "point": "The hook point, when vectors."},
    doc="Written before 2026-09-15 by `direction/similarity` as `{cosine}` or `{names, cosines, norms, pairs}`; "
        "those objects stay as stored and are read by the same fields.",
)
