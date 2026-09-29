from __future__ import annotations

from mechbench_compute.lexicon._base import Kind, Metric
from mechbench_compute.lexicon.values import F, TOP, TRACKED

KIND = Kind(
    "logits/distribution",
    "A summary of a next-token distribution: its entropy, the most likely tokens, and the tokens the caller asked about.",
    fields={"entropy_bits": F("number", "The distribution's entropy in bits."),
            "top": TOP, "tracked": TRACKED},
    required=("entropy_bits", "top"),
    doc="Every op that reads a next-token distribution produces this shape or a kind that extends it: "
        "the same `top` and `tracked`, spelled once. Two reads compare over the union of the tokens "
        "they carry, the mass neither names counted as one last bucket.",
    metrics=(
        Metric("jensen-shannon", "distance", True,
               "The Jensen–Shannon distance: the square root of the divergence in bits, in [0, 1]."),
        Metric("hellinger", "distance", True, "The Hellinger distance, in [0, 1]."),
        Metric("total-variation", "distance", True,
               "Half the L1 distance between the two: the largest difference in the probability of any event."),
        Metric("kl", "distance", False,
               "KL(p ‖ q) in bits, the row against the column. Not symmetric, so a tree refuses it."),
    ),
)
