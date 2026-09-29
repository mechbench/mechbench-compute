from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import COORDS, F, ID

KIND = Kind(
    "weights/parameter",
    "One of the model's learned tensors, read as data: where it sits in the module tree, its shape, and what its numbers are like.",
    extends="records/record",
    fields={"id": ID, "coords": COORDS,
            "shape": F("array", "The tensor's shape.", items={"type": "integer"}),
            "n": F("integer", "How many numbers it holds."),
            "dtype": F("string", "The dtype it is stored in (`bfloat16`, `float32`)."),
            "frobenius": F("number", "‖W‖_F — the tensor's magnitude."),
            "mean": F("number", "Mean of its values."),
            "std": F("number", "Standard deviation of its values."),
            "max_abs": F("number", "Largest magnitude in it, where an outlier channel shows."),
            "sparsity": F("number", "Share of its values that are exactly zero."),
            "singular_values": F("array", "The top singular values, largest first — only when `spectrum` asked for them.", items={"type": "number"}),
            "spectral": F("number", "σ₁, when the spectrum was computed."),
            "effective_rank": F("number", "exp(H(p)) over the spectrum, when it was computed: how many directions the tensor really uses."),
            "values": F("array", "The tensor itself, flattened — only when `values` asked for it, and only under the ceiling.", items={"type": "number"})},
    required=("id", "shape", "n", "frobenius"),
    key=("id",),
    header={"model": "The model's wire form — which weights these are.",
            "captured": "`{parameters, values, of}` — how many tensors this node read, how many numbers that is, and how many the model has."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="What `weights/capture` produces: one item per parameter tensor, `id` and `coords.module` naming it in the model's "
        "own tree (`layers.12.self_attn.q_proj.weight`). `coords` carry `layer`, `container`, `projection` and "
        "`parameter` where the name has them, so a reading groups by layer or by projection the way an activation "
        "capture groups by layer. The reduced forms are the default: the spectrum costs an SVD per tensor and the "
        "values are the tensor itself, so each is asked for.",
)
