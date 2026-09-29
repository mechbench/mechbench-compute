from __future__ import annotations

from mechbench_compute.lexicon._base import Kind

KIND = Kind(
    "activations/divergence",
    "For a matched pair, 1 − cosine between the two residual streams at every (layer, position).",
    extends="activations/grid",
    required=("id", "axes", "measures"),
    key=("id",),
    header={"point": "The residual compared.", "layers": "The layers, in row order."},
    doc="Axes `[layer, position]`; measure `divergence`; `tokens` are prompt `a`'s.",
)
