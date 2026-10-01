from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.dictionaries.describe_dictionary import describe_dictionary
from mechbench_compute.dictionaries.encode_feature import encode_feature
from mechbench_compute.dictionaries.encode_features import encode_features
from mechbench_compute.dictionaries.read_dictionary_weights import read_dictionary_weights
from mechbench_compute.dictionaries.read_feature import read_feature
from mechbench_compute.dictionaries.resolve_feature import resolve_feature
from mechbench_compute.distill import render
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.report_own_top1 import report_own_top1
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interp.resolve_target import resolve_target
from mechbench_compute.interp.answer import encode_answer
from mechbench_compute.lexicon._base import In, Op, Output, P, Resume

OP = Op(
    name="logits/attribute",
    needs=frozenset({"model.forward"}),
    resume=Resume("restart"),
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

A tracked answer is looked for with and without a leading space, but a
logit belongs to one token and two logits do not add, so the decomposition
is of the spelling the model gives more probability on this prompt.
`target` (and `contrast`) name the spelling decomposed; `variants` (and
`contrast_variants`) list every spelling with its probability, so a reader
sees how much of the answer the other spelling carries.

Because additivity only holds over the whole stream, `layers` must be
`"all"`.

### Attributing to features

A dictionary on the `dictionary` port splits the logit a second way, by
feature. At the last position the activation where the dictionary reads
is encoded, as `dictionary/encode` does it, and each feature that fires
contributes its activation times the direct logit attribution of its
decoder row: the row read through the unembedding, folded with the final
norm's scale as the components are. Each row then carries `features`,
the `top_features` largest contributions by size, and `reconstruction`,
which says how the attribution of the whole activation there (`stream`)
divides into the features' sum, the decoder bias's share (`b_dec`) and
what the dictionary does not reconstruct (`error`). It is the direct
effect of what the feature writes, as if the unembedding read it from
where it is written; what later layers make of it is not counted.

`feature: {"index": 3071}` attributes that one feature instead, firing
or not. The dictionary must write the residual stream (`resid_pre`,
`resid_post`, `attn_out` or `mlp_out`), because the unembedding reads
nothing else. The header's `feature` (or `dictionary`) names the
dictionary by its content hash.
""",
    inputs=(
        In("records", "records/record",
           "The prompts, one per record (`user`, `prompt` or `text`). A "
           "record may carry its own `tracked`."
           " A model reads each record as [its kind](/kinds/records/record/) says: `user` in the chat template, `prompt` and `text` raw.", many=True),
        In("dictionary", "direction/dictionary",
           "A dictionary to attribute the logit to, feature by feature, from "
           "`dictionary/load`; `feature.dictionary` names one in place of it.",
           required=False),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('logits/attribution', collection=True, doc="One grid per record over the axis `[component]`, in the order the header's `components` names the pieces (`embed`, `L0`, `L1`, …): `measures.contribution`, the `target` and `contrast` tokens (each the spelling the model prefers, with `variants` and `contrast_variants` listing every spelling as `{token, p, logp}`), the `additivity` check (`summed`, `true_logit`, `residual`), `per_head` when `per_head_layers` was set — each listed layer's contribution split by attention head — and, with a dictionary, `features` (each `{index, activation, dla, contribution}`) and `reconstruction` (`{stream, features, b_dec, error}`, over the whole dictionary only). The header then carries `dictionary`, or `feature` (`{dictionary, index}`), each dictionary as `{kind, hash, derivation, reads, width, source}`."),
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
        P("feature", "object",
          "One dictionary feature to attribute the logit to, in place of the "
          "dictionary's top contributors.", None, fields=(
              P("dictionary", "json",
                "The dictionary, usually a stored one (`{\"$ref\": …}`); or it "
                "arrives on the node's `dictionary` port.", None),
              P("index", "int", "The feature's index in the dictionary."),
          )),
        P("top_features", "int",
          "With a dictionary: how many of the largest feature contributions "
          "each row keeps, by size.",
          10),
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is "
          "the target — the logit being decomposed. Each answer is looked for "
          "with and without a leading space, and the spelling the model gives "
          "more probability on the prompt is the one decomposed. A "
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
        model, records, params, dictionary=inputs.get("dictionary"),
        on_item=ctx.on_item, on_start=ctx.on_start)


def attribute_logits(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    dictionary: Mapping[str, Any] | None = None,
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
    basis = read_basis(params, dictionary, model.arch.d_model)
    if basis is not None:
        interventions.append(Cap.at([basis["name"]]))
    rows: list[dict[str, Any]] = []
    for record in records:
        r = render(model, record)
        ids = r.array
        result = model.run(ids, interventions=interventions)
        lp = read_last_logp(result.logits)
        answer, tracked = resolve_target(model, record, params, lp)
        others = [a for a in tracked.values() if a.ids != answer.ids]
        contrast = record.get("contrast")
        canswer = (encode_answer(model.tokenizer, str(contrast)).anchored(lp) if contrast
                   else (others[0] if others else None))
        tok = answer.preferred
        ctok = canswer.preferred if canswer is not None else None

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
        cap = model.architecture.attribution_unembed(model._model).softcap
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
        by_feature = (attribute_features(model, basis, result.cache, targets, apply_ln=apply_ln,
                                         ln_scale=ln_scale, top=int(params.get("top_features", 10)))
                      if basis is not None else {})
        rows.append(S.grid(
            record.get("id"), ["component"],
            {"contribution": [round(float(x), 4) for x in contrib]},
            coords=record.get("coords"),
            target=S.token(model.tokenizer, tok),
            variants=answer.variants(model.tokenizer, lp),
            contrast=S.token(model.tokenizer, ctok) if ctok is not None else None,
            contrast_variants=(canswer.variants(model.tokenizer, lp)
                               if canswer is not None else None),
            template="chat" if r.chat else "raw",
            **report_own_top1(model, answer, lp),
            per_head=per_head or None,
            additivity={
                "summed": round(summed, 3),
                "true_logit": round(true_logit, 3),
                "residual": round(summed - true_logit, 3),
            },
            **by_feature))
        if on_item:
            on_item()
    return load_kinds().collection(
        "logits/attribution", rows,
        apply_ln=apply_ln,
        layers=layers,
        n_off_top1=sum(1 for r in rows if "own_top1" in r),
        components=["embed", *[f"L{i}" for i in layers]],
        **(basis["header"] if basis is not None else {}),
        description=(
            "Direct logit attribution: each component's contribution to "
            "the target logit (embedding first, then every layer's "
            "delta), norm-folded so the bars sum to the model's true "
            "final logit — each row carries its own additivity residual."
        ),
    )


RESIDUAL_POINTS = ("resid_pre", "resid_post", "attn_out", "mlp_out")


def read_basis(params: Mapping[str, Any], dictionary: Mapping[str, Any] | None,
               d_model: int) -> dict[str, Any] | None:
    feature = params.get("feature")
    if feature is None and dictionary is None:
        return None
    if feature is not None:
        dictionary, index = resolve_feature(feature, dictionary)
        one = read_feature(dictionary, index)
        reads, writes = one["reads"], one["writes"]
        header = {"feature": {"dictionary": describe_dictionary(dictionary), "index": index}}
    else:
        one = None
        weights = read_dictionary_weights(dictionary)
        reads = weights["space"]
        writes = dict((weights["header"].get("writes") or [reads])[0])
        header = {"dictionary": describe_dictionary(dictionary)}
    if writes.get("point") not in RESIDUAL_POINTS or int(writes.get("d") or d_model) != d_model:
        raise ValueError(
            f"the dictionary writes {writes.get('point')}, {writes.get('d')} wide; a feature's logit "
            f"attribution reads its decoder row through the unembedding, which takes the residual "
            f"stream ({', '.join(RESIDUAL_POINTS)}, {d_model} wide)")
    return {"name": f"blocks.{int(reads['layer'])}.{reads['point']}", "one": one,
            "weights": None if one is not None else weights, "header": header}


def attribute_features(model, basis: Mapping[str, Any], cache, targets: Sequence[int], *,
                       apply_ln: bool, ln_scale: np.ndarray, top: int) -> dict[str, Any]:
    from mechbench_compute import attribution

    x = np.array(mx.array(cache[basis["name"]]).astype(mx.float32))[0, -1:]

    def dla(rows: np.ndarray) -> np.ndarray:
        attrs = attribution.logit_attrs(model, rows[:, None, :], targets,
                                        apply_ln=apply_ln, ln_scale=ln_scale)
        return attrs[:, 0] if len(targets) == 1 else attrs[:, 0] - attrs[:, 1]

    def entry(i: int, value: float, d: float) -> dict[str, Any]:
        return {"index": int(i), "activation": round(value, 4), "dla": round(d, 4),
                "contribution": round(value * d, 4)}

    one = basis["one"]
    if one is not None:
        value = float(encode_feature(x, one)[0])
        return {"features": [entry(one["index"], value, float(dla(one["vector"][None])[0]))]}
    weights = basis["weights"]
    f = encode_features(x, weights)[0]
    active = np.flatnonzero(f)
    d_active = dla(weights["w_dec"][active]) if active.size else np.zeros(0)
    contrib = f[active].astype(np.float64) * d_active
    order = sorted(range(active.size), key=lambda j: (-abs(contrib[j]), int(active[j])))[:top]
    stream, b_dec = (float(v) for v in dla(np.stack([x[0], weights["b_dec"]])))
    total = float(contrib.sum())
    return {"features": [entry(active[j], float(f[active[j]]), float(d_active[j])) for j in order],
            "reconstruction": {"stream": round(stream, 4), "features": round(total, 4),
                               "b_dec": round(b_dec, 4), "error": round(stream - total - b_dec, 4)}}
