from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.directions.coerce_array import coerce_array
from mechbench_compute.directions.read_space import read_space
from mechbench_compute.lexicon._base import In, Op, Output, P

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
    inputs=(
        In("direction", "direction/vector", "The direction to read."),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
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
    model = ctx.model(params.get("model"))
    d = inputs.get("direction")
    return unembed_direction(model, d, top_k=int(params.get("top_k", 10)))


def unembed_direction(model, d: Mapping[str, Any], *, top_k: int = 10) -> dict[str, Any]:
    u = coerce_array(d)
    out: dict[str, Any] = {"kind": "direction/vocab", "space": read_space(d), "top_k": int(top_k)}
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        probs = np.asarray(model.decoded_distribution(sign * u), dtype=np.float64)
        logp = np.log(np.clip(probs, 1e-300, None))
        out[name] = S.distribution(logp, model.tokenizer, top_k=top_k)
    return out
