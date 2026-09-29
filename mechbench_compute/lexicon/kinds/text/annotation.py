from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "text/annotation",
    "One value anchored to a span of tokens in a document of a collection.",
    fields={"anchor": F("object", "`{item_id, token_start, token_end}` — where in which document."),
            "value": F("number", "The value at that span.")},
    required=("anchor", "value"),
    key=("anchor",),
    header={"name": "A label for the layer.", "description": "Free text beside the name.",
            "collection": "The stored collection the annotations are over.",
            "value_type": "`numeric` or `categorical`.",
            "required_fidelity": "The fidelity the collection must have been kept at."},
    doc="An annotation layer sits beside a document collection rather than inside it: the collection is left "
        "as stored and the layer points into it by document id and token span. `text/score` writes one, a "
        "surprisal per token; a viewer draws the layer over the text it annotates.",
)
