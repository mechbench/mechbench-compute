from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "direction/vector",
    "A unit vector in a model's activation space, with its derivation: how it was made, from what, on which model.",
    extends="activations/vector",
    fields={"unit": F("boolean", "Whether the vector is unit length."),
            "derivation": F("object", "`{method, sources, model, axis?, positive?, negative?, …}` — how it was made.")},
    required=("space", "vector", "derivation"),
    key=("id", "space"),
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="`norm` is the magnitude before normalisation, which some readings use.",
)
