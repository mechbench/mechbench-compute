from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.api import (
    MAX_VECTOR_FLOATS,
    In,
    Op,
    Output,
    P,
    Resume,
    add_to_span,
    check_written,
    collection,
    item_kind_of,
    items_of,
    positions,
    read_answer,
    read_last_logp,
    read_mask,
    read_record_coords,
    render,
    resolve_layers,
    resolve_outcomes,
    resolve_target,
    shapes,
)

METRICS = ("logit", "margin", "entropy_outcomes", "mass_outcomes")

OUTCOME_METRICS = ("entropy_outcomes", "mass_outcomes")

POINTS = ("resid_pre", "resid_post", "attn_out", "mlp_out", "gate_out")

WRITES = ("attn_out", "mlp_out", "gate_out")

OP = Op(
    name="activations/differentiate",
    needs=frozenset({"model.forward", "model.backward"}),
    resume=Resume("restart"),
    summary=(
        "Take the gradient of one number read at the model's decision (a target's logit, its margin over "
        "a contrast, or the entropy or mass on a record's outcomes) with respect to the activations at a "
        "point, per dimension and position, in one backward pass per prompt."
    ),
    description="""\
For each record the block runs the prompt forward once with the activation
at `point` exposed to differentiation in each of `layers`, reads `metric`
from the logits at the decision position (the prompt's last position, where
the next token is chosen), and runs one backward pass. The gradient is the
first-order sensitivity of the metric at the activations this prompt
produced, how fast the metric moves as one coordinate moves a little with
everything else held, and not an ablation: removing or replacing an
activation is a finite change, and a nonlinear model may answer it
differently.

Each layer and position gives one `activations/vector` in the shape
`activations/capture` emits, so `geometry/compare` and the `direction/*`
operations that take vectors read it unchanged. `vector` is the gradient,
`norm` its length, `position` and `token` where it was taken.
Gradients are small numbers on a large model, so they are kept to six
significant figures rather than to a fixed number of places.

### The metric

| `metric` | What is differentiated |
|---|---|
| `logit` | the target's logit |
| `margin` | the target's logit minus the contrast's |
| `entropy_outcomes` | the entropy in bits of the next-token distribution renormalised over the record's `outcomes` |
| `mass_outcomes` | the probability on the record's `outcomes` together |

The target is the first tracked token (a record's own `tracked`, `target`
or `outcomes` first, then the `tracked` param), or the model's own top-1
when none is named; the contrast is the record's `contrast`, or the second
tracked token. A token named by its text is the spelling the model gives
more probability on the prompt, with or without a leading space, as
`logits/attribute` decomposes it; a token named by id, `{"id": 9079}`, is
that token on every model. An outcome is matched by its first token with
and without a leading space, keeping the spellings whose token is the
whole outcome; a record without `outcomes` is refused by its id under the
outcome metrics.

`logit` and `margin` are read before any final softcap, the logits
`logits/attribute` decomposes, so a token in the cap's saturation still has
a gradient; the outcome metrics are read from the distribution the model
samples, after the cap. On an architecture with a cap (Gemma 4) the
header's `softcap` names it.

### Reading the gradient

`top: k` adds two rankings to every vector, the largest first, each entry
`{dim, value, share}` with `share` the entry's share of that vector's
squared norm: `top`, the gradient's own largest coordinates, and
`top_product`, the largest coordinates of the gradient times the
activation, coordinate by coordinate. That product is what the coordinate
adds to the metric to first order, so setting the coordinate to zero would
lower the metric by about that much: an estimate, not the effect of doing
so. A coordinate the model holds large, such as a massive activation, can
lead `top_product` with a small gradient.

`mask` names coordinates to read the derivative along, as an operator's
`mask` in `intervene/apply` names what it acts on: a list of dimensions, a
direction, a frame of several (a list of them, or a `direction/vector`
collection), or `"direction"` for the node's `direction` port. Each vector
then carries `along`, one number per coordinate of the mask in its order:
at a named dimension, the gradient's coordinate there; along a direction,
`∇·u` with `u` the direction's `vector` as stored (unit length unless the
direction kept its own scale), the metric's change per unit of `u` added,
which is its derivative with respect to the coordinate an operator's `x`
holds in that direction.

### Cost

One forward and one backward pass per record, however many layers and
positions are read. The backward pass runs from the decision position down
to the earliest layer named, so later layers are cheaper, and it holds that
span's activations in memory for the length of the prompt, as training on
one example does. The model's weights are not changed. The block refuses
more than two million values in all.

The variable of differentiation is a zero added at the point, which leaves
the forward pass the model's own. Where the backward pass crosses an MLP,
MLX computes the MLP's compiled activation by its separate operations,
which can move a float32 forward by a few units in the last place; with
only the last layer named it crosses none, and `value` is a plain read's
bit for bit.
""",
    inputs=(
        In("records", "records/record",
           "The prompts (`user`, `prompt` or `text`), each optionally with `coords`, its own `tracked`, "
           "a `contrast` for `margin`, and `outcomes` for the outcome metrics."
           " A model reads each record as [its kind](/kinds/records/record/) says: `user` in the chat "
           "template, `prompt` and `text` raw.", many=True),
        In("direction", "direction/vector",
           "A direction, or a frame as a `direction/vector` collection, that is the `mask` when the node "
           "names none or names `\"direction\"`.",
           required=False),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('activations/vector', collection=True, doc='One item per record per layer per position: `{id, coords, space, vector, norm, token, position}` as a capture emits them, `vector` the gradient of the metric there and `norm` its length, to six significant figures; `value`, the metric itself; `target` (and `contrast` under `margin`), the tokens differentiated; with `top`, `top` and `top_product`, each `[{dim, value, share}]`; with a mask, `along`. The header carries `model`, `metric`, `point`, `layers`, `positions`, `d_model`, `top`, `mask` (the dimensions, or each direction by its provenance) and, on an architecture with a final softcap, `softcap`.'),
    params=(
        P("metric", "string",
          "What is differentiated, read at the decision position: `logit` (the target's logit), "
          "`margin` (the target's logit minus the contrast's), `entropy_outcomes` (the entropy in bits "
          "of the distribution renormalised over the record's `outcomes`) or `mass_outcomes` (the "
          "probability on them together). `logit` and `margin` are read before any final softcap.",
          "logit", choices=METRICS),
        P("point", "string",
          "Where the gradient is taken: the residual stream entering a layer (`resid_pre`) or leaving it "
          "(`resid_post`), or a write into it, `attn_out`, `mlp_out`, or `gate_out` on a checkpoint "
          "with a per-layer input gate. A write the checkpoint does not make is refused by name with "
          "`POINT_ABSENT`, as `activations/capture` refuses it.",
          "resid_post", choices=POINTS, value="point"),
        P("layers", "list[int] | \"all\"",
          "Which layers, each its own vector from the same backward pass.",
          "all"),
        P("positions", "selector",
          "Which positions' activations, each its own vector: `\"last\"` (the decision position), "
          "`\"all\"`, a list of indices (negative from the end), `{\"tokens\": [...]}`, `{\"range\": "
          "[a, b]}`, `{\"after\": n}` or `\"subject\"`. The metric is read at the last position "
          "whatever this names.",
          "last"),
        P("top", "int",
          "How many of the largest coordinates each vector lists in `top` (by the gradient) and "
          "`top_product` (by the gradient times the activation); 0 lists none.",
          0),
        P("mask", "list[int] | \"direction\" | json",
          "Coordinates to read the derivative along: dimensions, a direction, a frame of several, or "
          "`\"direction\"` for the node's `direction` port. Each vector then carries `along`.",
          None),
        P("tracked", "map[string, string | object]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is the target and the second, "
          "unless a record names its `contrast`, the contrast. A token named by text is the spelling the "
          "model prefers on the prompt; one named by id, `{\"answer\": {\"id\": 9079}}`, is that token on "
          "every model. A record's own `tracked` takes precedence; with none named, the model's own "
          "top-1 prediction for the prompt is the target.",
          None, fields=(
              P("id", "int", "A token named by id: its id in the model's vocabulary."),
              P("text", "string", "The token's text, checked against the model's vocabulary "
                "when given.", None),
          )),
    ),
    example={
        "model": {"$param": "model"},
        "metric": "margin",
        "layers": [20],
        "top": 10,
        "tracked": {"answer": " Paris", "rival": " London"},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    records = items_of(inputs.get("records") or [])
    return differentiate_activations(model, records, params, direction=inputs.get("direction"),
                                     on_item=ctx.on_item, on_start=ctx.on_start)


def differentiate_activations(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    direction: Any = None,
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    metric = str(params.get("metric", "logit"))
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}: one of {', '.join(METRICS)}")
    point = str(params.get("point", "resid_post"))
    check_point(model, point)
    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
    selector = params.get("positions", "last")
    k = read_top(params.get("top", 0))
    width = int(model.arch.d_model)
    mask, mask_header = read_mask_param(params.get("mask"), direction, width, point)
    if not records:
        raise ValueError("activations/differentiate needs at least one record")
    prepared = [prepare_record(model, record, selector) for record in records]
    total = sum(len(at) for _, at in prepared) * len(layers) * width
    if total > MAX_VECTOR_FLOATS:
        raise ValueError(
            f"{len(records)} records × {len(layers)} layers × their positions × {width} dims = {total} "
            f"values exceeds the {MAX_VECTOR_FLOATS} cap: name fewer layers, positions or records")
    unembed = model.architecture.attribution_unembed(model._model)
    if on_start:
        on_start(len(records))
    names = [f"blocks.{layer}.{point}" for layer in layers]
    mid = shapes.model_id_of(model)
    rows: list[dict[str, Any]] = []
    for record, (rendered, at) in zip(records, prepared, strict=True):
        add_to_span(tokens_in=len(rendered.ids))
        found = differentiate_record(model, record, rendered.array, names, at, metric, params,
                                     unembed if unembed.softcap is not None else None)
        coords = read_record_coords(record, params)
        for layer, grads, acts in zip(layers, found["grads"], found["acts"], strict=True):
            sp = shapes.space(model=mid, layer=layer, point=point, d=width)
            for j, pos in enumerate(at):
                rows.append(build_row(model, record, sp, coords, pos, rendered.ids[pos],
                                      grads[j], acts[j], found, k, mask))
        if on_item:
            on_item()
    return collection(
        "activations/vector", rows,
        model=mid, metric=metric, point=point, layers=layers, positions=selector, d_model=width,
        top=k or None, mask=mask_header, softcap=unembed.softcap,
        description=(
            f"The gradient of the {metric} at the decision position with respect to {point}, one "
            "backward pass per record: the first-order sensitivity at the prompt's own activations, "
            "not an ablation."),
    )


def check_point(model, point: str) -> None:
    if point not in POINTS:
        raise ValueError(f"unknown point {point!r}: the gradient is taken at {', '.join(POINTS)}, the "
                         "residual stream and the writes into it")
    if point in WRITES:
        check_written(model, point)


def read_top(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"`top` counts coordinates, 0 or more; not {value!r}")
    return value


def read_mask_param(value: Any, port: Any, width: int, point: str) -> tuple[Any, Any]:
    if port is not None:
        if value is not None and value != "direction":
            raise ValueError("a direction arrived on the `direction` port and the node names its own "
                             "`mask`: one mask, the port's or the param's")
        value = port
    if value is None:
        return None, None
    mask = read_mask(value)
    mask.check(width, point)
    if mask.dims is not None:
        return mask, list(mask.dims)
    if isinstance(value, Mapping) and item_kind_of(value) is not None:
        found = items_of(value)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        found = list(value)
    else:
        found = [value]
    return mask, [shapes.direction_ref(d) for d in found]


def prepare_record(model, record: Mapping[str, Any], selector: Any) -> tuple[Any, list[int]]:
    rendered = render(model, record)
    n = len(rendered.ids)
    at = list(dict.fromkeys(positions.resolve(selector, n, tokens=rendered.tokens(model.tokenizer),
                                              record=record, prompt_len=rendered.prompt_len)))
    if not at:
        raise ValueError(f"record {record.get('id')!r}: positions {selector!r} names none of its "
                         f"{n} positions")
    return rendered, at


def differentiate_record(model, record: Mapping[str, Any], ids, names: Sequence[str], at: Sequence[int],
                         metric: str, params: Mapping[str, Any], unembed: Any) -> dict[str, Any]:
    outcomes = resolve_outcomes(model, record) if metric in OUTCOME_METRICS else None
    precap = unembed if outcomes is None else None
    shape = (1, int(ids.shape[-1]), int(model.arch.d_model))
    chosen: dict[str, int] = {}

    def read_value(deltas):
        hooks = {n: (lambda act, info, d=deltas[n]: act + d.astype(act.dtype)) for n in names}
        res = model.run(ids, hooks=hooks, capture=[*names, *(["final_norm"] if precap is not None else [])])
        row = res.logits[0, -1, :].astype(mx.float32)
        if outcomes is not None:
            return read_outcome_metric(metric, row, outcomes), [res.cache[n] for n in names]
        chosen.update(choose_tokens(model, record, params, read_last_logp(res.logits), metric))
        if precap is not None:
            row = precap.project(res.cache["final_norm"][0, -1:, :])[0].astype(mx.float32)
        value = row[chosen["target"]] if metric == "logit" else row[chosen["target"]] - row[chosen["contrast"]]
        return value, [res.cache[n] for n in names]

    (value, acts), grads = mx.value_and_grad(read_value)({n: mx.zeros(shape, dtype=mx.float32) for n in names})
    add_to_span(backwards=1)
    idx = mx.array(list(at), dtype=mx.int32)
    picked_grads = [mx.take(grads[n][0], idx, axis=0) for n in names]
    picked_acts = [mx.take(a[0], idx, axis=0).astype(mx.float32) for a in acts]
    mx.eval(value, picked_grads, picked_acts)
    return {"value": float(value), "grads": [np.array(g).astype(np.float64) for g in picked_grads],
            "acts": [np.array(a).astype(np.float64) for a in picked_acts], **chosen}


def choose_tokens(model, record: Mapping[str, Any], params: Mapping[str, Any], lp: np.ndarray,
                  metric: str) -> dict[str, int]:
    answer, tracked = resolve_target(model, record, params, lp, by_id=True)
    if metric == "logit":
        return {"target": int(answer.preferred)}
    contrast = record.get("contrast")
    other = (read_answer(model.tokenizer, contrast, by_id=True).anchored(lp) if contrast
             else next((a for a in tracked.values() if a.ids != answer.ids), None))
    if other is None:
        raise ValueError(f"record {record.get('id')!r}: `margin` is the target's logit minus a contrast's, "
                         "and the record names no `contrast` and tracks no second token")
    return {"target": int(answer.preferred), "contrast": int(other.preferred)}


def read_outcome_metric(metric: str, row, outcomes: Sequence[Any]):
    each = mx.stack([mx.logsumexp(mx.take(row, mx.array(list(o.ids), dtype=mx.int32))) for o in outcomes])
    if metric == "mass_outcomes":
        return mx.sum(mx.exp(each - mx.logsumexp(row)))
    logq = each - mx.logsumexp(each)
    return -mx.sum(mx.exp(logq) * logq) / math.log(2)


def build_row(model, record: Mapping[str, Any], sp: Mapping[str, Any], coords: Mapping[str, Any],
              pos: int, tid: int, grad: np.ndarray, act: np.ndarray, found: Mapping[str, Any], k: int,
              mask: Any) -> dict[str, Any]:
    tok = model.tokenizer
    row = shapes.vector(
        grad, sp, id=record.get("id"), coords=coords, token=shapes.token(tok, tid), position=int(pos),
        value=round_significant(found["value"]),
        target=shapes.token(tok, found["target"]) if "target" in found else None,
        contrast=shapes.token(tok, found["contrast"]) if "contrast" in found else None,
        top=rank_coordinates(grad, k), top_product=rank_coordinates(grad * act, k),
        along=read_along(mask, grad))
    row["vector"] = [round_significant(float(v)) for v in grad]
    row["norm"] = round_significant(float(np.linalg.norm(grad)))
    return row


def rank_coordinates(v: np.ndarray, k: int) -> list[dict[str, Any]] | None:
    if k <= 0:
        return None
    sq = v * v
    total = float(sq.sum())
    order = np.argsort(-np.abs(v), kind="stable")[:k]
    return [{"dim": int(i), "value": round_significant(float(v[i])),
             "share": math.floor(float(sq[i]) / total * 1e6) / 1e6 if total > 0 else 0.0} for i in order]


def read_along(mask: Any, grad: np.ndarray) -> list[float] | None:
    if mask is None:
        return None
    along = grad[mask.dims] if mask.dims is not None else mask.basis @ grad
    return [round_significant(float(v)) for v in along]


def round_significant(x: float) -> float:
    return float(f"{x:.6g}")
