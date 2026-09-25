from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import shapes as S
from mechbench_compute.distill import render
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interp.resolve_target import resolve_target
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P

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

A tracked answer is the set of its spellings with and without a leading
space, since which one the model says depends on what comes before it:
its log-probability is the log of the two spellings' summed probability,
and its rank is the better of the two ranks. `target` names the spelling
the model prefers at its final output, and `variants` lists each spelling
with its probability there.

The map answers: where in the sequence, and at what depth, does the answer
become visible?
""",
    inputs=(In("records", "records/record",
               "The prompts, one per record; a record's prompt is its `user`, "
               "`prompt` or `text` field. A record may carry its own `tracked`."
               " A model reads each record as [its kind](/kinds/records/record/) says: `user` in the chat template, `prompt` and `text` raw.",
               many=True), In("adapter", "adapter/lora",
                         "A LoRA adapter to fuse on top of the model for this node only — "
                         "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
                         "reference, or a stored adapter. Fuses last, on top of any "
                         "adapters the model reference itself carries; `adapter_scale` "
                         "scales this one.",
                         required=False)),
    output=Output('logits/lens', collection=True, doc="One grid per record over axes `[layer, position]`: `measures.logprob` (of the target's spellings together) and `measures.rank` (the better spelling's; 0 is the top readout), `tokens`, the `target` token (the spelling the model prefers at its output) and `variants` (each spelling as `{token, p, logp}` at the output)."),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to run over.",
          "all"),
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is "
          "the target — the answer being watched for. Each answer is followed "
          "with and without a leading space, whichever way it is written: its "
          "probability is the two spellings' sum, its rank the better one's. A "
          "record's own `tracked` takes precedence; with none named, the model's "
          "own top-1 prediction for that prompt is the target, and a target that "
          "differs from it is reported beside it.",
          None),
    ),
    example={
        "model": {"$param": "model"},
        "tracked": {"answer": " Paris"},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return scan_positions(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


def scan_positions(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    from mechbench_compute import lens

    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("lens/positions needs at least one condition")
    if on_start:
        on_start(len(records))

    cap = Capture.residual(layers, point="post")
    rows: list[dict[str, Any]] = []
    for record in records:
        ids = render(model, record).array
        result = model.run(ids, interventions=[cap])
        base_lp = read_last_logp(result.logits)
        answer, _ = resolve_target(model, record, params, base_lp)
        ranks, logprobs = lens.logit_lens_per_position(
            model, result.cache, answer, layers=layers)
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids).reshape(-1)]
        rows.append(S.grid(
            record.get("id"), ["layer", "position"],
            {"logprob": [[round(float(x), 4) for x in r] for r in logprobs],
             "rank": [[int(x) for x in r] for r in ranks]},
            tokens=tokens, coords=record.get("coords"),
            target=S.token(model.tokenizer, answer.preferred),
            variants=answer.variants(model.tokenizer, base_lp)))
        if on_item:
            on_item()
    return load_kinds().collection(
        "logits/lens", rows,
        layers=layers,
        description=(
            "Logit-lens readout of the target token at every (layer, "
            "position): log p and rank of the target when each layer's "
            "residual is projected straight through the unembedding."
        ),
    )
