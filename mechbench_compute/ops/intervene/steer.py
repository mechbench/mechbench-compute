from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as P
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.distill import render
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.collect_tracked_ids import collect_tracked_ids
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="intervene/steer",
    requires="mlx-local",
    summary=(
        "Build a steering direction from labelled residual vectors (one "
        "label's centroid minus another's), add it to the residual stream "
        "at one layer and position, and sweep its strength while reading the "
        "next-token distribution."
    ),
    description="""\
The direction comes from *data* flowing through the graph: a collection of
`activations/vector` whose items are grouped on a coordinate. At the
injection layer, the block takes the mean vector of the items whose
`direction.axis` coordinate is `direction.positive`, subtracts the mean of
those at `direction.negative`, and adds the result — scaled by each `alpha`
in turn — to the residual stream of every prompt at the chosen position.
Alpha 0 is the built-in control.

For anything beyond one direction at one layer and position, use
`intervene/apply`, of which this is a special case.
""",
    inputs=(
        In("records", "records/record",
           "The prompts to steer (`user`, `prompt` or `text`); a record may "
           "carry its own `position` and `tracked`.", many=True),
        In("vectors", "activations/vector",
           "The labelled vectors the direction is built from, with items at "
           "the injection `layer`.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('intervene/readout', collection=True, doc="One item per record per alpha: `id`, `coords`, `factor` (the alpha), `entropy_bits`, `top` (the most likely next tokens, each `{token, p, logp}`) and `tracked`. The header's `direction` reports the axis and the two values, the direction's norm and how many vectors went into each centroid; `sweep` lists the alphas."),
    params=(
        P("layer", "int",
          "The layer whose residual stream the direction is added to. Items "
          "at this layer must exist in the vectors collection."),
        P("direction", "object",
          "Which groups define the direction: `{\"axis\": \"register\", "
          "\"positive\": \"formal\", \"negative\": \"casual\"}` — the "
          "centroid of the items whose `axis` coordinate is `positive` minus "
          "the centroid of those at `negative`. `axis` defaults to `label`.", fields=(
              P("axis", "string", "The coordinate the groups are read on; `label` reads the older field too.",
                "label"),
              P("positive", "string | float", "The `axis` value of the items the direction points toward."),
              P("negative", "string | float", "The `axis` value of the items it points away from."),
          )),
        P("alphas", "list[float]",
          "The strengths to sweep. Each prompt is run once per alpha; "
          "alpha `0` is the untouched control.",
          [-8.0, -4.0, 0.0, 4.0, 8.0]),
        P("position", "selector",
          "The one token position the direction is added at: `\"last\"`, "
          "`\"all\"`, a list of indices (negative from the end), `{\"tokens\": "
          "[...]}`, `{\"range\": [a, b]}`, `{\"after\": n}`, `\"subject\"` or "
          "`\"generated\"`, resolving to a single position. A record's own "
          "`position` takes precedence.",
          "last"),
        P("top_k", "int",
          "How many of the most likely next tokens to record per row.",
          5),
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
        "layer": 12,
        "direction": {"axis": "register", "positive": "formal", "negative": "casual"},
        "alphas": [-4.0, 0.0, 4.0, 8.0],
        "tracked": {"hedge": " certainly"},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}, "vectors": {"$ref": {"bench": "you/lab/vectors"}}},
)


def run(ctx, inputs, params):
    """intervene/steer — a data-armed residual injection with an alpha
    sweep."""

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return steer_inject(
        model, records, params, inputs=inputs,
        on_item=ctx.on_item, on_start=ctx.on_start)


def steer_inject(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    inputs: Mapping[str, Any] | None = None,
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Steering as a block: build a direction from a labeled vector
    collection (centroid of `positive` minus centroid of `negative` at
    the injection layer) and ADD it to each eval prompt's residual
    stream at (layer, position), sweeping alpha. The readout is the
    final-position top-k under each alpha — alpha 0 is the built-in
    control.

    The direction comes from DATA flowing through the graph, not from a
    hardcoded vector: the same `activations/capture` that measures
    geometry also arms the intervention.
    """
    from mechbench_compute.interventions import Patch

    layer = params.get("layer")
    if not isinstance(layer, int):
        raise TypeError("steer/inject needs an integer `layer` to inject at")
    if not 0 <= layer < model.arch.n_layers:
        raise ValueError(
            f"layer {layer} out of range (n_layers={model.arch.n_layers})")
    alphas = [float(a) for a in params.get("alphas", [-8.0, -4.0, 0.0, 4.0, 8.0])]
    top_k = int(params.get("top_k", 5))
    direction = params.get("direction") or {}
    axis = str(direction.get("axis") or "label")
    pos_label = direction.get("positive")
    neg_label = direction.get("negative")
    if not pos_label or not neg_label:
        raise ValueError(
            "intervene/steer needs direction: {axis?: <coordinate>, "
            "positive: <value>, negative: <value>} naming groups in the "
            "vectors collection")

    vectors = (inputs or {}).get("vectors")
    if not isinstance(vectors, Mapping) or load_kinds().item_kind_of(vectors) != "activations/vector":
        raise ValueError(
            "intervene/steer needs a collection of activations/vector on "
            "its `vectors` port — the same block that measures geometry "
            "arms the intervention")
    rows_at = [r for r in load_kinds().items_of(vectors) if S.layer_of(r) == layer]
    pos = np.array([r["vector"] for r in rows_at if str(S.label_of(r, axis)) == str(pos_label)],
                   dtype=np.float32)
    neg = np.array([r["vector"] for r in rows_at if str(S.label_of(r, axis)) == str(neg_label)],
                   dtype=np.float32)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            f"the vectors collection has no items at layer {layer} with "
            f"{axis}={pos_label!r}/{neg_label!r} — capture that layer "
            "in activations/capture first")
    dvec = pos.mean(axis=0) - neg.mean(axis=0)
    dnorm = float(np.linalg.norm(dvec))

    if not records:
        raise ValueError("steer/inject needs at least one eval prompt")
    if on_start:
        on_start(len(records) * len(alphas))

    out_rows: list[dict[str, Any]] = []
    value = mx.array(dvec)
    for record in records:
        r = render(model, record)
        ids = r.array
        seq = len(r.ids)
        position = record.get("position", params.get("position", "last"))
        pos_idx = POS.one(position, seq, tokens=r.tokens(model.tokenizer),
                          record=record, prompt_len=r.prompt_len)
        tracked = collect_tracked_ids(model, record, tracked=params.get("tracked"))
        for alpha in alphas:
            interventions = (
                [] if alpha == 0.0
                else [Patch.add(layer, pos_idx, value, alpha=alpha)]
            )
            lp = read_last_logp(model.run(ids, interventions=interventions).logits)
            out_rows.append({
                "id": record.get("id"),
                "coords": dict(record.get("coords") or {}),
                "factor": alpha,
                **S.distribution(lp, model.tokenizer, top_k=top_k, tracked=tracked),
            })
            if on_item:
                on_item()
    return load_kinds().collection(
        "intervene/readout", out_rows,
        layer=layer,
        sweep={"strength": alphas},
        readout="decision",
        direction={
            "axis": axis,
            "positive": pos_label,
            "negative": neg_label,
            "norm": round(dnorm, 3),
            "n_positive": len(pos),
            "n_negative": len(neg),
        },
        description=(
            f"Residual injection at L{layer}: centroid({pos_label}) − "
            f"centroid({neg_label}), scaled by alpha, added at the "
            "chosen position. Alpha 0 is the control row."
        ),
    )
