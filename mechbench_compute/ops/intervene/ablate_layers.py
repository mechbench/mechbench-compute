from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import shapes as S
from mechbench_compute.distill import render
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.report_own_top1 import report_own_top1
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interp.resolve_target import resolve_target
from mechbench_compute.interventions import Ablate
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="intervene/ablate-layers",
    requires="mlx-local",
    summary=(
        "Remove one layer's contribution at a time and measure how much the "
        "target token's log-probability drops — which layers the answer "
        "depends on."
    ),
    description="""\
For each record, the block runs the prompt once untouched to get the target
token's baseline log-probability, then once per layer with the named
`point`(s) of that layer zeroed, and records the difference. A more negative
`delta_logp` means the layer was carrying more of the answer.

Zeroing a sub-layer's output leaves the residual stream unchanged by it;
zeroing both `attn_out` and `mlp_out` — the default — removes the whole
layer's contribution. Because records render the chat-shaped way, the
sweep can be taken at a decision point inside an assistant turn (a record
with a `prefill`), where `logits/read` reads.
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
    output=Output('intervene/ablation', collection=True, doc="Per record, one item per layer: `id`, `layer`, `delta_logp`. The header's `conditions` carry each record's untouched read (`{id, target, baseline_logp}`), and `aggregates.mean_delta` / `aggregates.median_delta` are per-layer across all records, in `layers` order."),
    params=(
        P("point", "string | list[string]",
          "The sub-layer output(s) zeroed at each layer: `\"attn_out\"`, "
          "`\"mlp_out\"`, or `\"gate_out\"` (the per-layer input gate on "
          "MatFormer-style models; other architectures refuse it), one or "
          "several. The default zeroes attention and MLP together — the "
          "whole layer.",
          ["attn_out", "mlp_out"], choices=("attn_out", "mlp_out", "gate_out"), value="point"),
        P("layers", "list[int] | \"all\"",
          "Which layers to run over.",
          "all"),
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is "
          "the target — the answer whose dependence on each layer is measured. "
          "Each is tokenized as a continuation of the rendered prompt: with a "
          "leading space after a raw prompt, without one after a chat template's "
          "assistant prefix. A record's own `tracked` takes precedence; with "
          "none named, the model's own top-1 prediction for that prompt is the "
          "target, and a target that differs from it is reported beside it.",
          None),
    ),
    example={
        "model": {"$param": "model"},
        "point": "mlp_out",
        "layers": "all",
        "tracked": {"answer": " Paris"},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    """Zero one layer — or one sublayer — at a time and measure the
    change in the target's log probability.

    The coarsest causal readout there is, and usually the first one
    worth running: it says WHERE in the stack the prediction is being
    built before any finer instrument is pointed at it.
    """

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return ablate_layers(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


#: The points `intervene/ablate-layers` can zero at a layer, by the point's
#: name. On a non-MatFormer model `gate_out` has no hook and the run
#: refuses with the arch's own error.
_ABLATE_AT: dict[str, Callable[[int], Any]] = {
    "attn_out": Ablate.attention,
    "mlp_out": Ablate.mlp,
    "gate_out": Ablate.side_channel,
}


def _resolve_ablation_points(spec: Any) -> list[str]:
    """The `point` param of `intervene/ablate-layers`: one name or a list of
    them, each a sub-layer output; the default is both, the whole
    layer's contribution."""
    if spec is None:
        return ["attn_out", "mlp_out"]
    names = [spec] if isinstance(spec, str) else [str(p) for p in spec]
    out = [hookpoints.normalize(n) for n in names]
    bad = [n for n in out if n not in _ABLATE_AT]
    if bad or not out:
        raise ValueError(
            f"intervene/ablate-layers zeroes a sub-layer output — one or more of "
            f"{sorted(_ABLATE_AT)} — not {bad or spec!r}")
    return out


def ablate_layers(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Per-layer ablation sweep: for each condition, zero the named
    `point`(s) of each layer in turn and measure Δ log p of the target —
    the first `tracked` token, or the baseline's top-1 when none is
    named."""
    points = _resolve_ablation_points(params.get("point"))
    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("ablate/layers needs at least one condition")
    if on_start:
        on_start(len(records) * (len(layers) + 1))

    def intervene(layer: int) -> list[Any]:
        # Zeroing both sub-layer outputs leaves the stream as it entered
        # the layer, which is the whole-layer skip.
        if set(points) == {"attn_out", "mlp_out"}:
            return [Ablate.layer(layer)]
        return [_ABLATE_AT[p](layer) for p in points]
    rows: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    damage_by_layer: dict[int, list[float]] = {i: [] for i in layers}
    for record in records:
        r = render(model, record)
        ids = r.array
        base_lp = read_last_logp(model.run(ids).logits)
        if on_item:
            on_item()
        tok, _ = resolve_target(model, record, params, base_lp)
        baseline = float(base_lp[tok])
        for layer in layers:
            lp = read_last_logp(model.run(ids, interventions=intervene(layer)).logits)
            delta = float(lp[tok]) - baseline
            damage_by_layer[layer].append(delta)
            rows.append({
                "id": record.get("id"),
                "layer": layer,
                "delta_logp": round(delta, 4),
            })
            if on_item:
                on_item()
        conditions.append({
            "id": record.get("id"),
            "target": S.token(model.tokenizer, tok),
            "baseline_logp": round(baseline, 4),
            # How the prompt reached the model. A record with only
            # `text` renders RAW — no chat template — and an instruct
            # model completing raw text answers with function words, so
            # a reader has to be able to see which envelope was used.
            "template": "chat" if r.chat else "raw",
            **report_own_top1(model, tok, base_lp),
        })

    return load_kinds().collection(
        "intervene/ablation", rows,
        points=points,
        layers=layers,
        n_conditions=len(records),
        # How many targets the model would not itself have said: a sweep
        # over the wrong spelling reads as a sweep, and this is the
        # number a reader checks before reading any Δ.
        n_off_top1=sum(1 for c in conditions if "own_top1" in c),
        conditions=conditions,
        aggregates={
            "mean_delta": [
                round(float(np.mean(damage_by_layer[i])), 4) for i in layers
            ],
            "median_delta": [
                round(float(np.median(damage_by_layer[i])), 4) for i in layers
            ],
        },
        description=(
            f"Δ log p of the target token when each layer's {'+'.join(points)} "
            "is zeroed; more negative = more load-bearing."
        ),
    )
