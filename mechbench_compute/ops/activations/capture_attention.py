from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.distill import render
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="activations/capture-attention",
    requires="mlx-local",
    summary=(
        "Record the attention weights of every head at the named layers — "
        "which earlier tokens each position attends to."
    ),
    description="""\
One forward pass per record captures the post-softmax attention weights at
each named layer. For each head the result is a square matrix over the
prompt's tokens: row = the attending position, column = the position
attended to.

`layers` must be given explicitly — the matrices are quadratic in prompt
length and there is one per head, so "all layers of a long prompt" is a
picture nobody asked for. The block refuses a capture that would exceed two
million values.
""",
    inputs=(
        In("records", "records/record",
           "The prompts, one per record; a record's prompt is its `user`, "
           "`prompt` or `text` field.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('activations/attention', collection=True, doc="One grid per record over axes `[layer, head, query, key]`: `measures.weight` is indexed in that order (row = the attending position, column = the attended-to position); `tokens` are the prompt's tokens."),
    params=(
        P("layers", "list[int]",
          "The layers whose attention to record. Must be named — `\"all\"` is "
          "refused."),
    ),
    example={
        "model": {"$param": "model"},
        "layers": [5, 6],
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    """activations/capture-attention — steps 05/06."""

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return capture_attention_patterns(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


#: Attention matrices are quadratic in sequence length; refuse a
#: capture that would emit more than this many floats.
MAX_ATTN_FLOATS = 2_000_000


def capture_attention_patterns(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Steps 05/06: post-softmax attention weights per head at chosen
    layers. Layers must be named explicitly — every layer of every
    head of a long prompt is a picture nobody asked for."""
    spec = params.get("layers")
    if spec in (None, "all"):
        raise ValueError(
            "attention/patterns needs an explicit layers list — "
            "attention weights are per-head and quadratic in prompt "
            "length, so name the layers you want to look at"
        )
    layers = resolve_layers(spec, model.arch.n_layers)
    if not records:
        raise ValueError("attention/patterns needs at least one condition")
    if on_start:
        on_start(len(records))

    cap = Capture.attn_weights(layers)
    rows: list[dict[str, Any]] = []
    total_floats = 0
    for record in records:
        ids = render(model, record).array
        result = model.run(ids, interventions=[cap])
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids).reshape(-1)]
        seq = len(tokens)
        total_floats += len(layers) * model.arch.n_heads * seq * seq
        if total_floats > MAX_ATTN_FLOATS:
            raise ValueError(
                f"attention capture would exceed {MAX_ATTN_FLOATS} floats "
                "— fewer layers, shorter prompts, or fewer conditions"
            )
        weight = []
        for layer in layers:
            w = result.cache[f"blocks.{layer}.attn.weights"]
            arr = np.array(w.astype(mx.float32))[0]  # [heads, L, S]
            weight.append([[[round(float(x), 4) for x in r] for r in h] for h in arr])
        rows.append(S.grid(
            record.get("id"), ["layer", "head", "query", "key"], {"weight": weight},
            tokens=tokens, coords=record.get("coords")))
        if on_item:
            on_item()
    return load_kinds().collection(
        "activations/attention", rows,
        n_heads=model.arch.n_heads,
        layers=layers,
        description=(
            "Post-softmax attention weights per head: row = the "
            "attending position, column = the attended-to position."
        ),
    )
