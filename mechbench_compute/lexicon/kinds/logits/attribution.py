from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F, TOKEN, VARIANTS

KIND = Kind(
    "logits/attribution",
    "A record's direct logit attribution: each component's contribution to the target logit, with the additivity check that says the pieces sum to the truth.",
    extends="activations/grid",
    fields={"target": TOKEN, "contrast": TOKEN, "variants": VARIANTS, "contrast_variants": VARIANTS,
            "per_head": F("array", "`{layer, contributions[head]}` for the layers split by head.", items={"type": "object"}),
            "additivity": F("object", "`{summed, true_logit, residual}` — the honesty number.")},
    required=("id", "axes", "measures", "target", "additivity"),
    key=("id",),
    header={"components": "The component names: `embed`, then `L0`, `L1`, …", "apply_ln": "Whether the final norm was folded in.",
            "layers": "The layers decomposed (all of them)."},
    doc="Axis `[component]`, in the header's `components` order; measure `contribution`.",
)
