from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import CELLS

KIND = Kind(
    "activations/divergence",
    "For a matched pair, 1 − cosine between the two residual streams at every (layer, position).",
    extends="activations/grid",
    fields={"cells": CELLS({"divergence": "1 − cosine between the two streams."}, token=True)},
    required=("id", "axes", "measures"),
    key=("id",),
    header={"point": "The residual compared.", "layers": "The layers, in row order."},
    doc="Axes `[layer, position]`; measure `divergence`; `tokens` are prompt `a`'s. `records/unnest field: cells` makes one record per (layer, position), `{address, point, layer, position, token, divergence}`, the point the header's and the position counted from the end.",
)
