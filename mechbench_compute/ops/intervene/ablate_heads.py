from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import shapes as S
from mechbench_compute.distill import render
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.report_own_top1 import report_own_top1
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interp.resolve_target import resolve_target
from mechbench_compute.interventions import Ablate
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="intervene/ablate-heads",
    requires="mlx-local",
    summary=(
        "Zero one attention head at a time across the chosen layers and "
        "measure the drop in the target token's log-probability — a "
        "layer × head map of which heads the answer runs through."
    ),
    description="""\
The head-level version of `intervene/ablate-layers`. For each record the baseline
log-probability of the target is taken once; then every (layer, head) pair in
turn has that head's output zeroed and the target re-read. The differences
are averaged over records into one matrix.

Cost is one forward pass per record per (layer, head): on a 30-layer,
16-head model that is 480 passes per record, so name the layers you care
about rather than all of them when the prompt set is large.
""",
    inputs=(In("records", "records/record",
               "The prompts, one per record; a record's prompt is its `user`, "
               "`prompt` or `text` field. A record may carry its own `tracked`.",
               many=True), In("adapter", "adapter/lora",
                         "A LoRA adapter to fuse on top of the model for this node only — "
                         "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
                         "reference, or a stored adapter. Fuses last, on top of any "
                         "adapters the model reference itself carries; `adapter_scale` "
                         "scales this one.",
                         required=False)),
    output=Output('intervene/heads', collection=False, doc="One grid over axes `[layer, head]`: `measures.mean_delta` is the mean Δ log‑p across records; `conditions` lists each record's `{id, target, baseline_logp}`; `layers` and `n_heads` give the axes."),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to run over.",
          "all"),
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is "
          "the target — the answer whose dependence on each head is measured. "
          "Each is tokenized as a continuation of the rendered prompt: with a "
          "leading space after a raw prompt, without one after a chat template's "
          "assistant prefix. A record's own `tracked` takes precedence; with "
          "none named, the model's own top-1 prediction for that prompt is the "
          "target, and a target that differs from it is reported beside it.",
          None),
    ),
    example={
        "model": {"$param": "model"},
        "layers": [10, 11, 12, 13, 14, 15],
        "tracked": {"answer": " Paris"},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return ablate_heads(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


def ablate_heads(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
    n_heads = model.arch.n_heads
    if not records:
        raise ValueError("ablate/heads needs at least one condition")
    if on_start:
        on_start(len(records) * (len(layers) + 1))

    sums = np.zeros((len(layers), n_heads), dtype=np.float64)
    metas: list[dict[str, Any]] = []
    for record in records:
        r = render(model, record)
        ids = r.array
        base_lp = read_last_logp(model.run(ids).logits)
        if on_item:
            on_item()
        tok, _ = resolve_target(model, record, params, base_lp)
        baseline = float(base_lp[tok])
        metas.append({
            "id": record.get("id"),
            "target": S.token(model.tokenizer, tok),
            "baseline_logp": round(baseline, 4),
            "template": "chat" if r.chat else "raw",
            **report_own_top1(model, tok, base_lp),
        })
        for li, layer in enumerate(layers):
            for head in range(n_heads):
                lp = read_last_logp(model.run(
                    ids, interventions=[Ablate.head(layer, head)]).logits)
                sums[li, head] += float(lp[tok]) - baseline
            if on_item:
                on_item()
    mean = sums / len(records)
    return {
        "kind": "intervene/heads",
        **S.grid("mean", ["layer", "head"],
                 {"mean_delta": [[round(float(x), 4) for x in row] for row in mean]}),
        "layers": layers,
        "n_heads": n_heads,
        "n_conditions": len(records),
        "n_off_top1": sum(1 for c in metas if "own_top1" in c),
        "conditions": metas,
        "description": (
            "Mean Δ log p of the target with each single head zeroed — "
            "rows are layers, columns are heads; dark cells are heads "
            "the answer runs through."
        ),
    }
