from __future__ import annotations

from mechbench_compute.lexicon._base import P, Kind, Metric
from mechbench_compute.lexicon.values import F, SPACE, TOKEN, VEC

KIND = Kind(
    "activations/vector",
    "One vector from a model's activation space, tagged with the space it lives in and what it was read from.",
    extends="records/record",
    fields={"space": SPACE, "vector": VEC,
            "norm": F("number", "The vector's magnitude."),
            "token": TOKEN,
            "n_pooled": F("integer", "How many positions went into it, when pooled.")},
    required=("space", "vector"),
    key=("id", "space"),
    header={"point": "The hook point read.", "source": "`resid`, `queries` or `keys`.",
            "position": "Which position, or `pooled`.", "pool": "The pooling, when pooled.",
            "layers": "The layers captured.", "d_model": "The vector width.",
            "model": "The model's wire form.",
            "skipped_empty": "Records dropped for having no text, when any.",
            "segments": "When made by `records/union`: the ports and how many each contributed."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="A grouping is a coordinate (`coords.genre`), never a `label` field; the ops that group take an `axis`. "
        "Two vectors compare only within one space; a metric refuses two that differ, naming both spaces.",
    metrics=(
        Metric("cosine", "similarity", True,
               "The cosine of the angle between the two, in [−1, 1]. With `center`, the collection's mean vector is "
               "subtracted first: transformer activations occupy a narrow cone around one dominant direction, and raw "
               "cosine measures that cone before it measures the items.",
               options=(P("center", "bool", "Subtract the collection's mean vector before comparing.", False),)),
        Metric("euclidean", "distance", True, "The straight-line distance between the two."),
        Metric("dot", "similarity", True, "The dot product, unnormalised; a tree over it uses max − dot as the distance."),
    ),
)
