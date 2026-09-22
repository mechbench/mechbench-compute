from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute._mlx import mx
from mechbench_compute.lexicon._base import In, Op, Output

OP = Op(
    name="text/score",
    requires="mlx-local",
    summary=(
        "Annotate every token of a trace-fidelity collection with its "
        "surprisal in bits under the model — how unexpected each token was."
    ),
    description="""\
For each item the block replays its stored token ids through the model and
records, for every position, −log₂ p(token | everything before it). Prompt
and envelope tokens are scored too, not only the generated body, so the
annotation covers the whole text and a reader can compare.

The collection must have been generated at `fidelity: "trace"`; a text-only
item has no token ids to replay and the block refuses it.
""",
    inputs=(
        In("collection", "text/document",
           "A trace-fidelity document collection, usually from `text/generate`; "
           "a stored one arrives as `{\"$ref\": …}`.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('text/annotation', collection=True, doc='One item per token: `{anchor: {item_id, token_start, token_end}, value}` with the surprisal in bits. The header names the `collection` scored and carries `value_type: "numeric"` and `required_fidelity: "trace"`.'),
    params=(),
    example={"model": {"$param": "model"}},
    example_inputs={"collection": {"$ref": {"bench": "you/lab/collection"}}},
)


def run(ctx, inputs, params):
    """Per-token surprisal (bits) under the scoring model, over a
    TRACE-fidelity collection. Values cover every position (prompt and
    envelope included): surprisal of token i given tokens < i. The
    output is a numeric annotation whose anchors are token spans, drawn
    as an overlay on the collection it scores."""
    import numpy as _np

    model = ctx.model(params.get("model"))
    coll = inputs.get("collection")
    coll_path = (ctx.input_paths or {}).get("collection", "")
    if coll is None:
        raise ValueError(
            "text/score needs a document collection on its `collection` "
            "port — by edge, or `{\"$fetch\": …}` under the node's inputs")
    items = lexicon.items_of(coll)
    if ctx.on_start:
        ctx.on_start(len(items))
    values = []
    for it in items:
        trace = it.get("trace")
        if not trace:
            raise ValueError(
                f"score: item {it.get('id')!r} has no trace — Score "
                "requires a trace-fidelity collection (set the "
                "Generate block's fidelity to 'trace')")
        ids = trace["token_ids"]
        h = model.trunk_hidden(mx.array([ids]))
        rows = model.head_logits(h[:, :-1, :]).astype(mx.float32)
        tgt = mx.array(ids[1:])
        lp = (mx.take_along_axis(rows[0], tgt[:, None], axis=-1)[:, 0]
              - mx.logsumexp(rows[0], axis=-1))
        surp = -_np.array(lp) / _np.log(2.0)
        for j, sv in enumerate(surp.tolist()):
            values.append({
                "anchor": {"item_id": it["id"],
                           "token_start": j + 1,
                           "token_end": j + 2},
                "value": round(float(sv), 3),
            })
        if ctx.on_item:
            ctx.on_item()
    return lexicon.collection(
        "text/annotation", values,
        name=params.get("name", "surprisal"),
        description=params.get(
            "description",
            "Per-token surprisal (bits); values cover every position "
            "including prompt and envelope tokens."),
        collection=coll_path,
        value_type="numeric",
        required_fidelity="trace")
