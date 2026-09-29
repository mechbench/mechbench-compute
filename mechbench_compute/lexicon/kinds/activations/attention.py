from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind(
    "activations/attention",
    "A record's attention weights at chosen layers: per head, a matrix over the prompt's positions.",
    extends="activations/grid",
    required=("id", "axes", "measures", "tokens"),
    key=("id",),
    header={"n_heads": "Heads per layer.", "layers": "The layers captured, in axis order."},
    doc="Axes `[layer, head, query, key]`; measure `weight` — row = the attending position, column = the attended-to position.",
)
