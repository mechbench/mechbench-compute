from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import COORDS, F, ID

KIND = Kind(
    "logits/funnel",
    "One layer of a record's commitment funnel: the next-token distribution when that layer's residual is read through the unembedding.",
    extends="logits/distribution",
    fields={"id": ID, "coords": COORDS, "layer": F("integer", "The layer read.")},
    required=("id", "layer", "entropy_bits", "top"),
    key=("id", "layer"),
    header={"name": "A label for the collection.", "description": "Free text beside the name.",
            "layers": "The layers read, in order.", "top_k": "How many tokens `top` holds."},
    renderer={"primitive": "table", "field_map": {"rows": "top"}},
    collection_renderer={"primitive": "series", "field_map": {"rows": "items", "x": "layer", "y": "entropy_bits", "label": "id"}},
    doc="Read a record's items in layer order and you see the funnel: entropy falling, one token taking over.",
)
