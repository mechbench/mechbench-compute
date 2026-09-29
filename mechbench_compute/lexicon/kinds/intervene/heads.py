from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "intervene/heads",
    "The mean change in a target token's log-probability with each single head zeroed: a layer × head grid over the records.",
    extends="activations/grid",
    fields={"layers": F("array", "The layers, in row order.", items={"type": "integer"}),
            "n_heads": F("integer", "Heads per layer."), "n_conditions": F("integer", "How many records."),
            "n_off_top1": F("integer", "How many conditions tracked a target the model would not itself have said."),
            "conditions": F("array", "Per record: `{id, target, variants, baseline_logp}`.", items={"type": "object"})},
    required=("id", "axes", "measures", "layers", "n_heads"),
    doc="Axes `[layer, head]`; measure `mean_delta`. One grid for the whole record set, id `mean`.",
)
