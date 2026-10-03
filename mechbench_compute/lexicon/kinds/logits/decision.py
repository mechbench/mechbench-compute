from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import COORDS, F, ID

KIND = Kind(
    "logits/decision",
    "The next-token distribution at a record's decision point, with the mass on each candidate outcome and an optional best-first expansion of complete outcomes.",
    extends="logits/distribution",
    fields={"id": ID, "coords": COORDS,
            "rollout": F("object", "The expanded complete outcomes, when a rollout was asked for.")},
    required=("id", "entropy_bits", "top"),
    key=("id",),
    header={"model": "The model read.", "top_k": "How many tokens `top` holds.",
            "softcap": "On a model whose final logits pass through a softcap, `c·tanh(x/c)`, the cap `c`; each tracked answer then also carries `logit`, `precap_logit` and `saturated`."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="What `logits/read` produces, one per condition, and what `eval/expect` judges. `tracked` holds "
        "each named token by the name the protocol gave it — the op's `tracked` param, or the record's own, "
        "which takes precedence — and the first is the target. `rollout`, when asked for, expands the most "
        "likely complete outcomes past the first token.",
)
