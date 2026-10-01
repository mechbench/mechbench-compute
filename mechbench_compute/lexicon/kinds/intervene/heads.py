from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "intervene/heads",
    "The mean change in a target token's log-probability, or another reading of the next token, with each single head zeroed: a layer × head grid over the records.",
    extends="activations/grid",
    fields={"layers": F("array", "The layers, in row order.", items={"type": "integer"}),
            "n_heads": F("integer", "Heads per layer."), "n_conditions": F("integer", "How many records."),
            "n_off_top1": F("integer", "How many conditions tracked a target the model would not itself have said."),
            "conditions": F("array", "Per record: `{id, target, variants, baseline_logp}`, and `baseline`, the "
                                     "metric on the untouched model, when the metric is not `logprob`.",
                            items={"type": "object"}),
            "metric": F("string", "`prob`, `logit`, `entropy`, `entropy_outcomes` or `mass_outcomes` when the "
                                  "grid reads one of them; absent for `logprob`.")},
    required=("id", "axes", "measures", "layers", "n_heads"),
    doc="Axes `[layer, head]`; measure `mean_delta`, the change in the `metric` (the target's log-probability "
        "when none is named). One grid for the whole record set, id `mean`. `intervene/prune` cuts it into "
        "an `intervene/circuit`.",
)
