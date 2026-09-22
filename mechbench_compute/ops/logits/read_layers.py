from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute._mlx import mx
from mechbench_compute.lexicon._base import In, Op, Output, P

_CHAT_RECORDS = In("records", "records/record",
                   "Chat-shaped records: `user` (required), `system` and "
                   "`prefill` (optional), an `id`, and optionally `coords`.",
                   many=True)


OP = Op(
    name="logits/read-layers",
    requires="mlx-local",
    summary=(
        "Logit lens at the decision point: for each chat-shaped record, the "
        "top-1 token, its probability and the entropy at every layer — the "
        "curve of how a model commits to an answer."
    ),
    description="""\
The record is rendered as a chat (`system`, `user`, and an optional
`prefill` that begins the assistant's turn), run once, and the residual
after every layer at the final position is projected through the
unembedding. Per layer the block records the most likely token, its
probability, and the distribution's entropy in bits.

Read a record's items in layer order and you see the commitment funnel:
entropy falling, one token taking over, at whichever depth this model
decides. A set of records renders as overlaid curves.
""",
    inputs=(
        _CHAT_RECORDS,
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('logits/funnel', collection=True, doc="One item per record per layer: `id`, `coords`, `layer`, and the distribution read through the unembedding at that layer — `entropy_bits`, `top` (the `top_k` most likely tokens, each `{token, p, logp}`) and `tracked`. The header carries `layers` and `top_k`."),
    params=(
        P("top_k", "int", "How many of the most likely tokens to record per layer.", 5),
        P("tracked", "map[string, string]",
          "Tokens to follow by name: `{\"yes\": \" Yes\", \"no\": \" No\"}` "
          "records each one's probability and log-probability under "
          "`tracked.<name>`. Tokenized as a continuation, so include the "
          "leading space where the model would. A record's own `tracked` "
          "field takes precedence.",
          None),
    ),
    example={
        "model": {"$param": "model"},
        "top_k": 5,
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/conditions"}}},
)


def run(ctx, inputs, params):
    """The logit-lens trajectory at the final position, per condition
    record: every layer's residual projected through the head,
    recording top-1 token, its probability, and entropy — the
    commitment-funnel instrument. Items are `logits/funnel`, so a
    collection renders as overlaid funnel curves."""
    import numpy as _np

    from mechbench_compute import Capture
    from mechbench_compute.distill import render

    model = ctx.model(params.get("model"))
    tok = model.tokenizer
    # By edge, or the common `records` param (a literal or a $fetch),
    # like every other model block: a block that read only the edge
    # would run over nothing when the prompts arrive by param.
    records = lexicon.items_of(inputs.get("records") or [])
    if not records:
        raise ValueError(
            "logits/read-layers: no records to run over — wire records to "
            "the `records` port")
    n_layers = len(model.lm.model.layers)
    top_k = int(params.get("top_k", 5))
    from mechbench_compute import shapes as S
    from mechbench_compute.interp import collect_tracked_ids

    if ctx.on_start:
        ctx.on_start(len(records))
    items = []
    for rec in records:
        r0 = render(model, rec)
        rendered, ids = r0.text, r0.ids
        r = model.run(
            mx.array([ids]),
            interventions=[Capture.residual(layers=range(n_layers))])
        tracked = collect_tracked_ids(model, rec, tracked=params.get("tracked"))
        for i in range(n_layers):
            row = model.project_to_logits(
                r.cache[f"blocks.{i}.resid_post"])[0, -1, :]
            z = _np.array(row.astype(mx.float32)).astype(_np.float64)
            logp = z - z.max() - _np.log(_np.exp(z - z.max()).sum())
            items.append({
                "id": rec["id"],
                "coords": dict(rec.get("coords", {})),
                "layer": i,
                **S.distribution(logp, tok, top_k=top_k, tracked=tracked),
            })
        if ctx.on_item:
            ctx.on_item()
    return lexicon.collection(
        "logits/funnel", items,
        name=params.get("name", "lens-trajectories"),
        description=params.get("description", ""),
        layers=list(range(n_layers)),
        top_k=top_k)
