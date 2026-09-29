from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "geometry/mst",
    "The minimum spanning tree over a group's pairwise distances, whatever metric produced them: the spread's scale and clumpiness, the bridges between clusters, and the edges.",
    fields={"group": F("string", "The group's name, from the similarity it was built on."),
            "layer": F("integer", "The layer, for a per-layer group."), "head": F("integer", "The head, for a per-head group."),
            "n": F("integer", "Items in the group."), "n_edges": F("integer", "Edges in the tree."),
            "mean": F("number", "Mean edge: the scale of the spread."), "variance": F("number", "Edge variance: the clumpiness."),
            "stdev": F("number", "Edge standard deviation."), "cv": F("number", "stdev / mean, scale-free."),
            "total": F("number", "Sum of edge weights."), "max": F("number", "Longest edge."), "min": F("number", "Shortest edge."),
            "bridge_threshold": F("number", "Edges above this count as bridges."), "bridges": F("integer", "Edges above the threshold."),
            "components_after_cut": F("integer", "Clusters left when the bridges are cut."),
            "ids": F("array", "The item ids, in edge-index order.", items={}), "labels": F("array", "Their labels.", items={}),
            "edges": F("array", "`[i, j, weight]` per edge, in the order the tree grew.", items={"type": "array"})},
    required=("n", "n_edges"),
    key=("group",),
    header={"name": "A label for the summary.", "metric": "The metric the distances came from.",
            "options": "The metric's options as applied.", "over": "The kind of the items compared.",
            "bridge_sigma": "The bridge threshold in standard deviations.", "axis": "The coordinate the labels read."},
)
