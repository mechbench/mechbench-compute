from __future__ import annotations

from mechbench_compute.lexicon._base import Draw, Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "geometry/alignment",
    "One alignment score per pair of layers between two sets of similarity matrices over the same records.",
    doc="Written by `geometry/align`. Each item compares layer `a_layer` of the first set with layer `b_layer` "
        "of the second. Linear CKA lies in 0 to 1 on centred Gram matrices; Spearman RSA lies in -1 to 1. "
        "A layer whose matrix is constant scores 0.",
    extends="records/record",
    fields={"a_layer": F("integer", "The layer in the first set."),
            "b_layer": F("integer", "The layer in the second set."),
            "score": F("number", "Linear CKA, or the Spearman correlation of the upper triangles, to six places.")},
    required=("a_layer", "b_layer", "score"),
    key=("a_layer", "b_layer"),
    header={"method": "`cka` or `rsa`.",
            "center": "Whether each Gram matrix was double-centred before CKA.",
            "a": "The first set's header: its metric, what it compared, and how.",
            "b": "The second set's header, in the same form."},
    speak="{header.method} alignment of layers {a_layer} and {b_layer}: {score}",
    draw=Draw(mark="heat", encoding={"x": "a_layer", "y": "b_layer", "value": "score"}),
)
