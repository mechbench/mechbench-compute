from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import shapes as S
from mechbench_compute.distill import render
from mechbench_compute.interp.k import _K
from mechbench_compute.interp.last_logp import _last_logp
from mechbench_compute.interp.resolve_layers import _resolve_layers
from mechbench_compute.interp.target_of import _target_of
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import Op, Output
from mechbench_compute.lexicon.model import _LAYERS_ALL, _PROMPTS, ADAPTER, _tracked

OP = Op(
    name="logits/scan",
    requires="mlx-local",
    summary=(
        "Logit lens over the whole prompt: at every (layer, position), how "
        "probable and how highly ranked the target token is when that "
        "residual is read straight through the unembedding."
    ),
    description="""\
One forward pass per record, capturing the residual after every requested
layer. Each captured vector, at each position, is projected through the
model's unembedding as if it were the final layer, and the target token's
log-probability and rank are read off. Rank 0 means the target is that
position's top readout.

The map answers: where in the sequence, and at what depth, does the answer
become visible?
""",
    inputs=(_PROMPTS, ADAPTER),
    output=Output('logits/lens', collection=True, doc="One grid per record over axes `[layer, position]`: `measures.logprob` and `measures.rank` (0 is the top readout), `tokens`, and the `target` token."),
    params=(
        _LAYERS_ALL,
        _tracked("the answer being watched for"),
    ),
    example={
        "model": {"$param": "model"},
        "tracked": {"answer": " Paris"},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    """Read the target's probability at every (layer, position), by
    projecting each mid-stack residual through the unembedding.

    The logit lens over a whole sequence rather than one point: where
    in the text, and how deep in the stack, does the answer become
    visible? "Visible at layer k" means decodable there, which is not
    the same as decided there. (Step 08's question as a block.)
    """

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return lens_positions(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


def lens_positions(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Step 08 as a block: project every layer's residual through the
    unembedding at every position and follow one target token — where
    in the sequence, and at what depth, does the answer become
    visible? Rank 0 means the target is that position's top readout."""
    from mechbench_compute import lens

    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("lens/positions needs at least one condition")
    if on_start:
        on_start(len(records))

    cap = Capture.residual(layers, point="post")
    rows: list[dict[str, Any]] = []
    for record in records:
        ids = render(model, record).array
        result = model.run(ids, interventions=[cap])
        tok, _ = _target_of(model, record, params, _last_logp(result.logits))
        ranks, logprobs = lens.logit_lens_per_position(
            model, result.cache, tok, layers=layers)
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids).reshape(-1)]
        rows.append(S.grid(
            record.get("id"), ["layer", "position"],
            {"logprob": [[round(float(x), 4) for x in r] for r in logprobs],
             "rank": [[int(x) for x in r] for r in ranks]},
            tokens=tokens, coords=record.get("coords"),
            target=S.token(model.tokenizer, tok)))
        if on_item:
            on_item()
    return _K().collection(
        "logits/lens", rows,
        layers=layers,
        description=(
            "Logit-lens readout of the target token at every (layer, "
            "position): log p and rank of the target when each layer's "
            "residual is projected straight through the unembedding."
        ),
    )
