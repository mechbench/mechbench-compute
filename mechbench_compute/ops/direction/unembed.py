from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute.directions.as_array import as_array
from mechbench_compute.directions.space_of import space_of
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.lexicon.model import ADAPTER

OP = Op(
    name="direction/unembed",
    requires="mlx-local",
    summary=(
        "Read a direction through the model's unembedding: the tokens it "
        "promotes and the tokens its negative promotes — what the axis "
        "'says' in vocabulary."
    ),
    description="""\
The direction (and its negative) is passed through the model's final norm
and unembedding as if it were a residual state, and the most probable tokens
in each sign are listed. The final norm is scale-invariant, so a unit
direction reads the same as any multiple of it.

The reading is only literal for directions at the residual stream; a
direction inside an attention block is not in the space the unembedding
reads.
""",
    inputs=(In("direction", "direction/vector", "The direction to read."), ADAPTER),
    output=(
        Output('direction/vocab', collection=False, doc="`space`, `top_k`, and `positive` and `negative` — each a distribution (`entropy_bits`, `top` as `{token, p, logp}`) of the unembedding applied to that sign.")
    ),
    params=(
        P("top_k", "int", "How many tokens to list per sign.", 10),
    ),
    example={"model": {"$param": "model"}, "top_k": 20},
    example_inputs={"direction": {"$ref": {"bench": "you/lab/axis"}}},
)


def run(ctx, inputs, params):
    """direction/unembed (task 000367): a direction
    through the unembedding — its top tokens in both signs."""

    model = ctx.model(params.get("model"))
    d = inputs.get("direction")
    return vocab_projection(model, d, top_k=int(params.get("top_k", 10)))


def vocab_projection(model, d: Mapping[str, Any], *, top_k: int = 10) -> dict[str, Any]:
    """What a direction 'says' in token space: the distribution the
    unembedding gives +d and −d (the final norm is scale-invariant, so a
    unit direction is as good as any multiple)."""
    u = as_array(d)
    out: dict[str, Any] = {"kind": "direction/vocab", "space": space_of(d), "top_k": int(top_k)}
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        probs = np.asarray(model.decoded_distribution(sign * u), dtype=np.float64)
        logp = np.log(np.clip(probs, 1e-300, None))
        out[name] = S.distribution(logp, model.tokenizer, top_k=top_k)
    return out
