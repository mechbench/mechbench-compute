from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import DIST, F, SPACE

KIND = Kind(
    "direction/vocab",
    "What a direction says in token space: the distribution the unembedding gives it and its negative.",
    fields={"space": SPACE, "top_k": F("integer", "How many tokens per sign."),
            "positive": DIST, "negative": DIST},
    required=("space", "positive", "negative"),
    doc="What `direction/unembed` produces: the direction pushed through the unembedding as if it were a final "
        "residual, and its negative likewise, each read as a next-token distribution. The tokens the positive "
        "side promotes are what the axis 'says'; the negative side is what it says when reversed. Two "
        "distributions, so the distribution metrics compare a direction's vocabulary with another's.",
)
