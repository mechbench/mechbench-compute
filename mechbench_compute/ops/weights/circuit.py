from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="weights/circuit",
    requires="mlx-local",
    summary=(
        "What a head does, read from its own weights and the vocabulary: "
        "the OV circuit it writes through, the QK circuit it looks with, "
        "and how much of each earlier head lands in what it reads."
    ),
    description="""\
Everything else in `intervene` and `activations` watches a model run. This
reads the weights, and runs nothing: no corpus, no forward pass, no
sampling. It is how a head is NAMED before it is patched — "L23H5 copies
the subject token", "L12H2 feeds L23H5's query" — which is the step a
circuit story usually skips and then cannot defend.

**`ov`** takes the top singular components of W_O·W_V, the map from what
the head attends to onto what it writes into the residual stream. Each
component pairs an input direction with an output direction, and both are
read through the embedding as tokens: `right` are the tokens that trigger
the component, `left` the tokens it then promotes. A copying head shows
the same tokens on both sides; an induction head shows the token that
FOLLOWED them.

**`qk`** does the same for W_Qᵀ·W_K: `left` are the query tokens a
component looks for, `right` the key tokens it matches. Position is not in
it — RoPE is applied to the activations, not the weights — so a head that
attends by position alone shows nothing here, which is itself the finding.

**`composition`** reads INTO one head (`head: {"layer": 23, "index": 5}`)
and scores every head of the earlier layers against it: ‖W_read·W_OV‖ over
‖W_read‖·‖W_OV‖, with `W_read` the destination's W_Q, W_K or W_V — Q-, K-
and V-composition (Elhage et al. 2021). A high score means what that head
writes reaches this head's query, key or value; it is a bound on the
influence, not a demonstration of it, and `intervene/path` is what
demonstrates. The scale is set by chance: read the scores against their own
median, not against zero.

Every row is an ordinary record, so `records/rank value: "strength"` finds
the strongest components and `records/select where: {"kind": "q"}` the
query-side composers.
""",
    inputs=(In("adapter", "adapter/lora",
               "A LoRA adapter to fuse on top of the model for this node only — "
               "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
               "reference, or a stored adapter. Fuses last, on top of any "
               "adapters the model reference itself carries; `adapter_scale` "
               "scales this one.",
               required=False),),
    output=Output('records/record', collection=True,
                  doc='For `ov` and `qk`, one item per (layer, head, component): `coords` (`layer`, `head`, `rank`, `circuit`), `strength` (the singular value), `left` and `right` (each `{token, score}`), `kv_group`. For `composition`, one per (source head, kind): `coords` (`layer`, `head`, `kind`, `into_layer`, `into_head`) and `score`. The header carries `circuit`, the model, and `into` or `components`/`top_k`.'),
    params=(
        P("circuit", "string",
          "`\"ov\"` (what the head writes, and what triggers it), `\"qk\"` "
          "(what it looks for, and what matches) or `\"composition\"` (how "
          "much of each earlier head reaches this one).",
          "ov", choices=("ov", "qk", "composition")),
        P("head", "object",
          "One head: `{\"layer\": 23, \"index\": 5}`. Required for "
          "`composition`, which reads into it; for `ov` and `qk` it names "
          "the single head to read, where `layers`/`heads` name a set.",
          None, fields=(
              P("layer", "int", "Its layer."),
              P("index", "int", "Its index within the layer.", 0),
          )),
        P("layers", "list[int]",
          "For `ov`/`qk`: which layers' heads to read, every layer by "
          "default. For `composition`: which earlier layers to score, all "
          "of them by default.",
          None),
        P("heads", "list[int]",
          "For `ov`/`qk`: which heads of each layer, all by default.", None),
        P("kinds", "list[string]",
          "For `composition`: which of the destination's inputs to score.",
          ["q", "k", "v"], choices=("q", "k", "v")),
        P("components", "int",
          "For `ov`/`qk`: how many singular components per head.", 3),
        P("top_k", "int",
          "For `ov`/`qk`: how many tokens to read off each direction.", 10),
    ),
    example={"model": {"$param": "model"}, "circuit": "ov",
             "head": {"layer": 23, "index": 5}, "components": 2},
)


def run(ctx, inputs, params):
    """weights/circuit: what a head does, read from its own weights
    and the vocabulary — the OV and QK circuits, and the
    composition of earlier heads with this one. No forward pass."""

    model = ctx.model(params.get("model"))
    return read_head_circuits(model, params, on_item=ctx.on_item,
                                     on_start=ctx.on_start)


def read_head_circuits(model, params: Mapping[str, Any] | None = None,
                       on_item=None, on_start=None) -> dict[str, Any]:
    """What a head does, read from its weights and the vocabulary — no
    forward pass, no corpus.

    `ov`: the top singular components of W_O·W_V, each an input
    direction (tokens that trigger it) paired with an output direction
    (tokens it then promotes). A copying head shows the same tokens on
    both sides.

    `qk`: the same for W_Qᵀ·W_K — the query tokens a component looks
    for, and the key tokens it matches. Positional terms are not in it:
    RoPE is applied to the activations, not the weights.

    `composition`: how much of what each earlier head WRITES lands in
    what this head READS, as Q-, K- or V-composition — which is how one
    names the heads that feed a head before patching any of them.
    """
    from mechbench_compute import head_weights as hw
    from mechbench_compute.lexicon import kinds as K

    params = dict(params or {})
    which = str(params.get("circuit", "ov"))
    if which not in ("ov", "qk", "composition"):
        raise ValueError(f"circuit is 'ov', 'qk' or 'composition', not {which!r}")
    n_layers, n_heads = model.arch.n_layers, model.arch.n_heads
    head = params.get("head")

    if which == "composition":
        if not isinstance(head, Mapping) or "layer" not in head:
            raise ValueError(
                'composition reads INTO one head: name it, `head: {"layer": 12, "index": 3}`')
        rows = hw.head_composition(
            model, int(head["layer"]), int(head.get("index", 0)),
            source_layers=params.get("layers"),
            kinds=tuple(params.get("kinds") or ("q", "k", "v")))
        if on_start:
            on_start(len(rows))
        if on_item:
            for _ in rows:
                on_item()
        return K.collection(
            "records/record", rows, circuit="composition",
            into={"layer": int(head["layer"]), "index": int(head.get("index", 0))},
            model=str(getattr(model, "model_id", "") or ""),
            description=(
                "Q-, K- and V-composition of each earlier head with this one: "
                "how much of what it writes lands in what this head reads."))

    if isinstance(head, Mapping):
        pairs = [(int(head["layer"]), int(head.get("index", 0)))]
    else:
        layers = ([int(x) for x in params["layers"]] if params.get("layers") is not None
                  else list(range(n_layers)))
        heads = ([int(x) for x in params["heads"]] if params.get("heads") is not None
                 else list(range(n_heads)))
        pairs = [(lay, h) for lay in layers for h in heads]
    for lay, h in pairs:
        if not 0 <= lay < n_layers or not 0 <= h < n_heads:
            raise ValueError(
                f"layer {lay} head {h} is outside this model ({n_layers} layers, "
                f"{n_heads} heads)")
    components = int(params.get("components", 3))
    top_k = int(params.get("top_k", 10))
    if on_start:
        on_start(len(pairs))

    def _tokens(pairs_):
        return [{"token": t, "score": round(float(v), 5)} for t, v in pairs_]

    analyse = hw.ov_circuit if which == "ov" else hw.qk_circuit
    rows = []
    for lay, h in pairs:
        got = analyse(model, lay, h, k_tokens=top_k, n_components=components)
        for comp in got.components:
            rows.append({
                "id": f"L{lay}H{h}:{comp.rank}",
                "coords": {"layer": lay, "head": h, "rank": comp.rank, "circuit": which},
                "strength": round(float(comp.strength), 5),
                # For `ov`: what it writes, and what triggers the write.
                # For `qk`: what the query looks for, and what the key
                # matches. The names say which side, not which meaning.
                "left": _tokens(comp.left_tokens),
                "right": _tokens(comp.right_tokens),
                "kv_group": got.kv_group,
            })
        if on_item:
            on_item()
    return K.collection(
        "records/record", rows, circuit=which, components=components, top_k=top_k,
        model=str(getattr(model, "model_id", "") or ""),
        description=(
            "The top singular components of each head's "
            + ("W_O·W_V: the tokens that trigger a write, and the tokens it writes."
               if which == "ov" else
               "W_Qᵀ·W_K: the query tokens a component looks for, and the keys it matches.")))
