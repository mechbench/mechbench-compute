from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.distill import render
from mechbench_compute.interp.k import _K
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.lexicon.model import ADAPTER

OP = Op(
    name="activations/examples",
    requires="mlx-local",
    summary=(
        "The corpus windows whose token most excites a direction or a "
        "neuron — what turns it on, in context — with the corpus never "
        "held."
    ),
    description="""\
The first thing anyone asks of a direction, a neuron or a feature is what
turns it on, and the answer is a handful of windows out of a corpus of any
size. Every record is run once, every token projected onto the direction
(or read off the neuron), and only the best `k` windows are kept: the
memory is `k × window`, not the corpus, so this is the op to point at a
hundred thousand tokens.

Name what to excite once: a `direction` on the port — whose own space says
which layer and point to read, so a probe from `direction/classify` needs
nothing further — or a `neuron` `{"layer": 14, "index": 2048}`, read at
`mlp.act` unless `point` says otherwise.

Each window carries `values` as well as `tokens` — every token's own
projection, not only the winner's — which is what `records/plot` draws as
a token strip.

`sign` chooses the end: `"high"` (the default), `"low"` — the tokens that
most oppose it, which is where a direction's meaning often becomes clear —
or `"both"`. The header's `over` carries the corpus's own moments (count,
mean, sd, min, max), so a window's value can be read against the field it
came from rather than as a bare number.
""",
    inputs=(
        In("records", "records/record", "The corpus to search.", many=True),
        In("direction", "direction/vector",
           "The direction to excite; its space says where to read. A "
           "collection carrying exactly one direction is that direction.",
           required=False),
        ADAPTER,
    ),
    output=Output('records/record', collection=True,
                  doc='One item per kept window: `value` (the projection at the exciting token), `token`, `text` (the window), `tokens` (its token strings), `values` (each of their projections, so `records/plot mark: "tokens"` colours the whole window), `hit` (the exciting token\'s index among them), `rank`, and `coords` with `record`, `position` and — under `sign: "both"` — `side`. The header carries `model`, `layer`, `point`, `window`, `sign`, `neuron` when one was named, and `over`: the corpus\'s `n_tokens`, `mean`, `sd`, `min`, `max`.'),
    params=(
        P("k", "int", "How many windows to keep at each end.", 10),
        P("window", "int", "How many tokens either side of the exciting one.", 8),
        P("sign", "string",
          "Which end: `\"high\"`, `\"low\"`, or `\"both\"` (which marks each "
          "window's `side`).",
          "high", choices=("high", "low", "both")),
        P("neuron", "object",
          "The neuron to excite, in place of a direction.", None, fields=(
              P("layer", "int", "Its layer."),
              P("index", "int", "Its index along the feature axis."),
          )),
        P("layer", "int",
          "Which layer to read, when the direction does not say.", None),
        P("point", "string",
          "Where to read: the direction's own point by default, `mlp.act` "
          "for a neuron.",
          None, value="point"),
    ),
    example={"model": {"$param": "model"}, "k": 5, "window": 6, "sign": "both"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/passages"}},
                    "direction": {"$ref": {"bench": "you/lab/surprise-axis"}}},
)


def run(ctx, inputs, params):
    """activations/examples (task 000615) — the corpus windows that
    most excite a direction or a neuron, the corpus never held."""

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return examples(model, records, params,
                           direction=inputs.get("direction"),
                           on_item=ctx.on_item, on_start=ctx.on_start)


def examples(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    direction: Mapping[str, Any] | None = None,
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """The corpus windows that most excite a direction or a neuron, with
    the corpus never held (task 000615).

    The first thing anyone asks of a direction, a neuron or a feature is
    "what turns it on" — and the answer is a handful of windows out of a
    corpus of any size. Every record is run once, every token projected
    onto the direction (or read off the neuron), and only the best `k`
    windows are kept: memory is k × window, not the corpus. What the
    whole corpus was like rides on the header as the moments of the
    projection, so a window's value can be read against the field it
    came from.
    """
    from mechbench_compute import directions as dirs

    neuron = params.get("neuron")
    if (direction is None) == (neuron is None):
        raise ValueError(
            "name what to excite: a `direction` on the port, or a `neuron` "
            "`{layer, index}` — one of them, not both")
    if neuron is not None:
        if not isinstance(neuron, Mapping) or "layer" not in neuron or "index" not in neuron:
            raise ValueError('`neuron` is `{"layer": 14, "index": 2048}`')
        layer, index = int(neuron["layer"]), int(neuron["index"])
        point = hookpoints.normalize(str(params.get("point", "mlp.act")))
        vec = None
    else:
        sp = dirs.space_of(direction)
        layer = params.get("layer", sp.get("layer"))
        if layer is None:
            raise ValueError("the direction names no layer; give `layer`")
        layer = int(layer)
        point = hookpoints.normalize(str(params.get("point") or sp.get("point") or "resid_post"))
        vec = dirs.as_array(direction)
        index = None
    k = int(params.get("k", 10))
    half = int(params.get("window", 8))
    sign = str(params.get("sign", "high"))
    if sign not in ("high", "low", "both"):
        raise ValueError(f"sign is 'high', 'low' or 'both', not {sign!r}")
    if not records:
        raise ValueError("examples needs at least one record")
    if on_start:
        on_start(len(records))

    name = point if layer is None else f"blocks.{layer}.{point}"
    keep_high: list[tuple[float, dict[str, Any]]] = []
    keep_low: list[tuple[float, dict[str, Any]]] = []
    n_tokens = 0
    total = total_sq = 0.0
    lo_all, hi_all = float("inf"), float("-inf")
    for record in records:
        r = render(model, record)
        res = model.run(r.array, interventions=[Capture.at([name])])
        act = res.cache[name][0].astype(mx.float32)     # [L, d] (or [L, heads, d])
        if vec is not None:
            values = np.array(mx.sum(act * mx.array(vec), axis=-1))
        else:
            values = np.array(act[:, index])
        toks = r.tokens(model.tokenizer)
        n_tokens += int(values.size)
        total += float(values.sum())
        total_sq += float(np.sum(values.astype(np.float64) ** 2))
        lo_all, hi_all = min(lo_all, float(values.min())), max(hi_all, float(values.max()))

        def window_at(pos: int, value: float) -> dict[str, Any]:
            a, b = max(0, pos - half), min(len(toks), pos + half + 1)
            return {"id": f"{record.get('id')}:{pos}",
                    "coords": {**(record.get("coords") or {}), "record": record.get("id"),
                               "position": int(pos)},
                    "value": round(float(value), 5),
                    "token": toks[pos],
                    "text": "".join(toks[a:b]),
                    "tokens": list(toks[a:b]),
                    # Every token of the window, not only the one that
                    # won it: a token strip (000616) colours them all,
                    # and the shape of the rise is the interesting part.
                    "values": [round(float(v), 5) for v in values[a:b]],
                    "hit": int(pos - a)}

        # Only this record's best few can enter the running top, so the
        # corpus is never sorted whole.
        if sign in ("high", "both"):
            for pos in np.argsort(-values)[:k]:
                keep_high.append((float(values[pos]), window_at(int(pos), values[pos])))
            keep_high = sorted(keep_high, key=lambda t: -t[0])[:k]
        if sign in ("low", "both"):
            for pos in np.argsort(values)[:k]:
                keep_low.append((float(values[pos]), window_at(int(pos), values[pos])))
            keep_low = sorted(keep_low, key=lambda t: t[0])[:k]
        if on_item:
            on_item()

    items = []
    for side, kept in (("high", keep_high), ("low", keep_low)):
        for rank, (_, w) in enumerate(kept):
            if sign == "both":
                w = {**w, "coords": {**w["coords"], "side": side}}
            items.append({**w, "rank": rank})
    mean = total / max(n_tokens, 1)
    var = max(total_sq / max(n_tokens, 1) - mean * mean, 0.0)
    return _K().collection(
        "records/record", items,
        model=S.model_id_of(model), point=point, layer=layer,
        **({"neuron": index} if index is not None else {}),
        window=half, sign=sign,
        # What the corpus was like, so a window's value reads against it.
        over={"n_tokens": n_tokens, "mean": round(mean, 5),
              "sd": round(float(np.sqrt(var)), 5),
              "min": round(lo_all, 5), "max": round(hi_all, 5)},
        description=(
            "The corpus windows whose token most excites "
            + (f"neuron {index} at layer {layer}" if index is not None
               else "the direction")
            + ", the exciting token marked by `hit` among the window's tokens."))
