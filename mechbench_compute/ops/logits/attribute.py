from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.distill import render
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.report_own_top1 import report_own_top1
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interp.resolve_target import resolve_target
from mechbench_compute.interp.encode_target_token import encode_target_token
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="logits/attribute",
    requires="mlx-local",
    summary=(
        "Split the target token's final logit into the additive contribution "
        "of the embedding and of every layer — direct logit attribution, with "
        "each row checking that the pieces sum to the truth."
    ),
    description="""\
The residual stream at the last position is the embedding plus each layer's
update. Each of those components is projected through the unembedding (folded
with the final norm's scale when `apply_ln` is on) to give its contribution
to the target's logit, so the contributions are exactly additive.

Every row also reports its **additivity residual**: the summed contributions
minus the model's true final logit for the target. A reader never has to take
the decomposition on faith — a residual far from zero says the decomposition
does not describe this model.

With two `tracked` tokens, the contributions are to the *difference* of
their logits (the first minus the second), which is usually the more
interpretable quantity.

Because additivity only holds over the whole stream, `layers` must be
`"all"`.
""",
    inputs=(
        In("records", "records/record",
           "The prompts, one per record (`user`, `prompt` or `text`). A "
           "record may carry its own `tracked`.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('logits/attribution', collection=True, doc="One grid per record over the axis `[component]`, in the order the header's `components` names the pieces (`embed`, `L0`, `L1`, …): `measures.contribution`, the `target` and `contrast` tokens, the `additivity` check (`summed`, `true_logit`, `residual`), and `per_head` when `per_head_layers` was set — each listed layer's contribution split by attention head."),
    params=(
        P("apply_ln", "bool",
          "Fold the final norm's scale into the unembedding so contributions "
          "are in the same units as the model's real logits. Turning it off "
          "gives the raw, un-normalised projection.",
          True),
        P("layers", "list[int] | \"all\"",
          "Must be `\"all\"`: the decomposition is only additive over the "
          "whole stream.",
          "all"),
        P("per_head_layers", "list[int]",
          "Layers at which to also split the attention contribution by "
          "head. Opt-in per layer because per-head outputs cost a slower "
          "attention path.",
          None),
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is "
          "the target — the logit being decomposed. Each is tokenized as a "
          "continuation of the rendered prompt: with a leading space after a raw "
          "prompt, without one after a chat template's assistant prefix. A "
          "record's own `tracked` takes precedence; with none named, the model's "
          "own top-1 prediction for that prompt is the target, and a target that "
          "differs from it is reported beside it.",
          None),
    ),
    example={
        "model": {"$param": "model"},
        "tracked": {"answer": " Paris"},
        "per_head_layers": [12, 13],
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return attribute_logits(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


def attribute_logits(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    from mechbench_compute import attribution

    apply_ln = bool(params.get("apply_ln", True))
    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
    if layers != list(range(model.arch.n_layers)):
        raise ValueError(
            "attribution/logits decomposes the WHOLE stream — additivity "
            "only holds over all layers, so `layers` must be \"all\""
        )
    if not records:
        raise ValueError("attribution/logits needs at least one condition")
    if on_start:
        on_start(len(records))

    from mechbench_compute.interventions import Capture as Cap

    per_head_layers = resolve_layers(
        params.get("per_head_layers"), model.arch.n_layers
    ) if params.get("per_head_layers") else []
    interventions = [
        Cap.residual(layers, point="post"),
        Cap.residual([0], point="pre"),
        Cap.final_norm_scale(),
    ]
    if per_head_layers:
        interventions.append(Cap.per_head_out(per_head_layers))
    rows: list[dict[str, Any]] = []
    for record in records:
        r = render(model, record)
        ids = r.array
        result = model.run(ids, interventions=interventions)
        lp = read_last_logp(result.logits)
        tok, tracked = resolve_target(model, record, params, lp)
        others = [t for t in tracked.values() if t != tok]
        contrast = record.get("contrast")
        ctok = (encode_target_token(model, str(contrast)) if contrast
                else (others[0] if others else None))

        acc = attribution.accumulated_resid(result.cache, include_pre=True)
        components = np.diff(acc, axis=0, prepend=np.zeros_like(acc[:1]))
        ln_scale = np.array(
            mx.array(result.cache["final_norm.scale"]).astype(mx.float32)
        ).reshape(-1)
        targets = [tok] if ctok is None else [tok, ctok]
        attrs = attribution.logit_attrs(
            model, components, targets,
            apply_ln=apply_ln, ln_scale=ln_scale)
        contrib = attrs[:, 0] if ctok is None else attrs[:, 0] - attrs[:, 1]

        last = result.logits[0, -1, :].astype(mx.float32)
        mx.eval(last)
        last_np = np.array(last, dtype=np.float64)
        cap = getattr(
            getattr(getattr(model, "_model", None), "language_model", None),
            "final_logit_softcapping", None)
        if cap:
            c = float(cap)
            last_np = c * np.arctanh(np.clip(last_np / c, -0.999999, 0.999999))
        true_logit = float(last_np[tok])
        if ctok is not None:
            true_logit -= float(last_np[ctok])
        summed = float(attrs.sum(axis=0)[0]) if ctok is None else float(
            (attrs[:, 0] - attrs[:, 1]).sum())
        per_head: list[dict[str, Any]] = []
        for hl in per_head_layers:
            hr = attribution.head_results(model, result.cache, hl)
            hattrs = attribution.logit_attrs(
                model, hr, targets, apply_ln=apply_ln, ln_scale=ln_scale)
            hc = (hattrs[:, 0] if ctok is None
                  else hattrs[:, 0] - hattrs[:, 1])
            per_head.append({
                "layer": hl,
                "contributions": [round(float(x), 4) for x in hc],
            })
        rows.append(S.grid(
            record.get("id"), ["component"],
            {"contribution": [round(float(x), 4) for x in contrib]},
            coords=record.get("coords"),
            target=S.token(model.tokenizer, tok),
            contrast=S.token(model.tokenizer, ctok) if ctok is not None else None,
            template="chat" if r.chat else "raw",
            **report_own_top1(model, tok, lp),
            per_head=per_head or None,
            additivity={
                "summed": round(summed, 3),
                "true_logit": round(true_logit, 3),
                "residual": round(summed - true_logit, 3),
            }))
        if on_item:
            on_item()
    return load_kinds().collection(
        "logits/attribution", rows,
        apply_ln=apply_ln,
        layers=layers,
        n_off_top1=sum(1 for r in rows if "own_top1" in r),
        components=["embed", *[f"L{i}" for i in layers]],
        description=(
            "Direct logit attribution: each component's contribution to "
            "the target logit (embedding first, then every layer's "
            "delta), norm-folded so the bars sum to the model's true "
            "final logit — each row carries its own additivity residual."
        ),
    )
