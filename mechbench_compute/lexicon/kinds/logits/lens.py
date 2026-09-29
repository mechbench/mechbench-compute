from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import TOKEN, VARIANTS

KIND = Kind(
    "logits/lens",
    "A record's logit lens over the whole prompt: the target token's log-probability and rank at every (layer, position).",
    extends="activations/grid",
    fields={"target": TOKEN, "variants": VARIANTS},
    required=("id", "axes", "measures", "tokens", "target"),
    key=("id",),
    header={"layers": "The layers read, in row order."},
    doc="Axes `[layer, position]`; measures `logprob` and `rank` (0 is the top readout).",
)
