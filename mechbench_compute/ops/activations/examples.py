from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.dictionaries.describe_dictionary import describe_dictionary
from mechbench_compute.dictionaries.encode_feature import encode_feature
from mechbench_compute.dictionaries.read_dictionary_activations import (
    read_dictionary_activations,
)
from mechbench_compute.dictionaries.read_feature import read_feature
from mechbench_compute.dictionaries.resolve_feature import resolve_feature
from mechbench_compute.dictionaries.resolve_features import resolve_features
from mechbench_compute.distill import render
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P, Resume
from mechbench_compute.seeds import derive as derive_seed

OP = Op(
    name="activations/examples",
    needs=frozenset({"model.forward"}),
    resume=Resume("restart"),
    summary=(
        "The corpus windows whose token most excites a direction, a "
        "neuron or a dictionary's feature — what turns it on, in context — "
        "with the corpus never held."
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
nothing further — a `neuron` `{"layer": 14, "index": 2048}`, read at
`mlp.act` unless `point` says otherwise, or a `feature` `{"index": 3071}`
of the dictionary on the `dictionary` port.

A feature is a neuron in another basis. Its value at a position is what
`dictionary/encode` gives it there: the activation at the point and layer
the dictionary reads, times the feature's encoder column, plus its
`b_enc`, through the dictionary's nonlinearity (under a JumpReLU, zero
unless it passes the feature's own threshold). The dictionary says where
to read, so `layer` and `point` are refused beside it. The
beginning-of-sequence token reads zero, because its activation is an
outlier dictionaries are not trained on and would win every ranking. The
header's `feature` names the dictionary by its content hash and the
feature by its index.

`features: [3, 71, 2048]` in place of `feature` reads the corpus once for
all of them: each record is run once, and every listed feature is encoded
from the same activations. The result is one strip per feature in one
collection, each window's `coords.feature` naming its feature, `k` windows
per feature (per end), the features in the order listed; a window's `id`
carries the feature after the record and position. The header's
`features` (`{dictionary, indices}`) stands where the single form's
`feature` does, and `over` is a list with one entry per feature. Each
feature's windows are the ones `feature` gives for it alone.

A direction or a neuron leaves the beginning-of-sequence position out
(`skip_bos`, as `dictionary/encode` does): its activation is an outlier
that wins any projection it touches. The position is then neither ranked,
nor counted in `over`, nor shown at the left edge of a window.

Each window carries `values` as well as `tokens` — every token's own
projection, not only the winner's — which is what `records/plot` draws as
a token strip.

`sign` chooses the end: `"high"` (the default), `"low"` — the tokens that
most oppose it, which is where a direction's meaning often becomes clear —
or `"both"`. `"random"` draws `k` windows uniformly, without replacement
and reproducibly under `seed`, from the positions where the feature fires
(any value but zero) — for a direction or a neuron, from every position
read — which is a held-out set rather than the strongest. `per_record`
caps how many windows any one record gives each end (or the draw) before
the `k` are kept, so one record that fires along its whole length does not
fill them. The header's `over` carries the corpus's own moments (count,
mean, sd, min, max), so a window's value can be read against the field it
came from rather than as a bare number.
""",
    inputs=(
        In("records", "records/record", "The corpus to search."
                                        " A model reads each record as [its kind](/kinds/records/record/) says: `user` in the chat template, `prompt` and `text` raw.", many=True),
        In("direction", "direction/vector",
           "The direction to excite; its space says where to read. A "
           "collection carrying exactly one direction is that direction.",
           required=False),
        In("dictionary", "direction/dictionary",
           "The dictionary that `feature` or `features` index, from `dictionary/load` or a stored one.",
           required=False),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('records/record', collection=True,
                  doc='One item per kept window: `value` (the projection at the exciting token), `token`, `text` (the window), `tokens` (its token strings), `values` (each of their projections, so `records/plot mark: "tokens"` colours the whole window), `hit` (the exciting token\'s index among them), `rank`, and `coords` with `record`, `position`, `feature` under `features`, and `side` under `sign: "both"`. Under `sign: "random"`, `rank` is the order of the draw. The header carries `model`, `layer`, `point`, `window`, `sign`, `per_record` when given, `seed` under `sign: "random"`, `skip_bos` for a direction or a neuron, `neuron` when one was named, `feature` (`{dictionary: {kind, hash, derivation, reads, width, source}, index}`) when one was, `features` (`{dictionary, indices}`) when several were, and `over`: the corpus\'s `n_tokens`, `mean`, `sd`, `min`, `max` — under `features`, a list of them, each with its `feature`.'),
    params=(
        P("k", "int", "How many windows to keep at each end, or to draw.", 10),
        P("window", "int", "How many tokens either side of the exciting one.", 8),
        P("sign", "string",
          "Which end: `\"high\"`, `\"low\"`, `\"both\"` (which marks each "
          "window's `side`), or `\"random\"`: `k` windows drawn under `seed` "
          "from where the feature fires, or from every position for a "
          "direction or a neuron.",
          "high", choices=("high", "low", "both", "random")),
        P("per_record", "int",
          "At most this many windows from any one record at each end, before the `k` are kept.",
          None),
        P("skip_bos", "bool",
          "For a direction or a neuron: leave out the first position when it is the "
          "beginning-of-sequence token. A feature reads it as zero either way.",
          True),
        P("neuron", "object",
          "The neuron to excite, in place of a direction.", None, fields=(
              P("layer", "int", "Its layer."),
              P("index", "int", "Its index along the feature axis."),
          )),
        P("feature", "object",
          "The feature of the dictionary on the `dictionary` port to excite, in place of a "
          "direction or a neuron.", None, fields=(
              P("index", "int", "The feature's index in the dictionary."),
          )),
        P("features", "list[int]",
          "Several features of the dictionary on the `dictionary` port, read in one pass over "
          "the corpus, in place of `feature`: one strip per feature, keyed by `coords.feature`.",
          None),
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
    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return find_top_examples(model, records, params,
                           direction=inputs.get("direction"),
                           dictionary=inputs.get("dictionary"),
                           on_item=ctx.on_item, on_start=ctx.on_start)


SIDES = {"high": ("high",), "low": ("low",), "both": ("high", "low"), "random": ("random",)}


def find_top_examples(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    direction: Mapping[str, Any] | None = None,
    dictionary: Mapping[str, Any] | None = None,
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    from mechbench_compute import directions as dirs

    neuron = params.get("neuron")
    feature = params.get("feature")
    features = params.get("features")
    if neuron is not None and (feature is not None or features is not None):
        raise ValueError(
            "`feature` and `neuron` are one address in two bases, a dictionary's "
            "and the model's own: name one of them, not both")
    if feature is not None and features is not None:
        raise ValueError("`feature` names one feature and `features` several: give one of them")
    if sum(x is not None for x in (direction, neuron, feature, features)) != 1:
        raise ValueError(
            "name what to excite: a `direction` on the port, a `neuron` `{layer, index}`, "
            "or a `feature` `{index}` or `features` `[…]` of the dictionary on the `dictionary` "
            "port — exactly one of them")
    feats = None
    index = None
    if feature is not None or features is not None:
        if feature is not None:
            dictionary, index = resolve_feature(feature, dictionary)
            indices = [index]
        else:
            dictionary, indices = resolve_features(features, dictionary)
        feats = [read_feature(dictionary, i) for i in indices]
        reads = feats[0]["reads"]
        if params.get("layer") is not None or params.get("point") is not None:
            raise ValueError(
                f"a feature is read where its dictionary reads, {reads['point']} at layer "
                f"{reads['layer']}: leave out `layer` and `point`")
        layer, point = int(reads["layer"]), str(reads["point"])
        vec = None
    elif neuron is not None:
        if not isinstance(neuron, Mapping) or "layer" not in neuron or "index" not in neuron:
            raise ValueError('`neuron` is `{"layer": 14, "index": 2048}`')
        layer, index = int(neuron["layer"]), int(neuron["index"])
        point = hookpoints.normalize(str(params.get("point", "mlp.act")))
        vec = None
    else:
        sp = dirs.read_space(direction)
        layer = params.get("layer", sp.get("layer"))
        if layer is None:
            raise ValueError("the direction names no layer; give `layer`")
        layer = int(layer)
        point = hookpoints.normalize(str(params.get("point") or sp.get("point") or "resid_post"))
        vec = dirs.coerce_array(direction)
    k = int(params.get("k", 10))
    half = int(params.get("window", 8))
    sign = str(params.get("sign", "high"))
    if sign not in SIDES:
        raise ValueError(f"sign is 'high', 'low', 'both' or 'random', not {sign!r}")
    per_record = params.get("per_record")
    if per_record is not None and (not isinstance(per_record, int) or isinstance(per_record, bool)
                                   or per_record < 1):
        raise ValueError(f"`per_record` is a whole number of windows, at least 1, not {per_record!r}")
    skip_bos = bool(params.get("skip_bos", True))
    seed = params.get("seed", 0)
    if not records:
        raise ValueError("examples needs at least one record")
    if on_start:
        on_start(len(records))

    sides = SIDES[sign]
    cap = k if per_record is None else min(k, per_record)
    keys = [None] if feats is None else [f["index"] for f in feats]
    strips = {key: Strip(sides) for key in keys}
    name = f"blocks.{layer}.{point}"
    bos = getattr(model.tokenizer, "bos_token_id", None)
    for i, record in enumerate(records):
        r = render(model, record)
        starts_with_bos = bos is not None and len(r.ids) and int(r.ids[0]) == bos
        first = 0
        if feats is not None:
            x = read_dictionary_activations(model, model.make_ids(r.ids), point, layer)
            rows = []
            for feat in feats:
                values = encode_feature(x, feat)
                if starts_with_bos:
                    values[0] = np.float32(0)
                rows.append(values)
        else:
            res = model.run(model.make_ids(r.ids), interventions=[Capture.at([name])])
            act = res.cache[name][0].astype(mx.float32)
            if vec is not None:
                rows = [np.array(mx.sum(act * mx.array(vec), axis=-1))]
            else:
                rows = [np.array(act[:, index])]
            if skip_bos and starts_with_bos:
                first = 1
        toks = r.tokens(model.tokenizer)
        for key, values in zip(keys, rows, strict=True):
            draw = None
            if sign == "random":
                draw = np.random.default_rng(derive_seed(seed, "activations/examples", key, i)).random(len(values))
            strips[key].take(record, toks, values, first=first, half=half, k=k, cap=cap, draw=draw,
                             fires_only=feats is not None)
        if on_item:
            on_item()

    items = []
    for key in keys:
        for side in sides:
            for rank, (_, w) in enumerate(strips[key].kept[side]):
                if features is not None:
                    w = {**w, "id": f"{w['id']}:{key}", "coords": {**w["coords"], "feature": key}}
                if sign == "both":
                    w = {**w, "coords": {**w["coords"], "side": side}}
                items.append({**w, "rank": rank})
    over = ([{"feature": key, **strips[key].moments()} for key in keys] if features is not None
            else strips[keys[0]].moments())
    if feats is not None:
        width, derivation = feats[0]["header"].get("width"), feats[0]["header"].get("derivation")
        target = (f"feature {index}" if features is None
                  else "each of features " + ", ".join(str(j) for j in keys))
        target += f" of the {width}-wide {derivation} at {point} layer {layer}"
    else:
        target = f"neuron {index} at layer {layer}" if index is not None else "the direction"
    return load_kinds().collection(
        "records/record", items,
        model=S.model_id_of(model), point=point, layer=layer,
        **({"neuron": index} if index is not None and feats is None else {}),
        **({"feature": {"dictionary": describe_dictionary(dictionary), "index": index}}
           if feature is not None else {}),
        **({"features": {"dictionary": describe_dictionary(dictionary), "indices": list(keys)}}
           if features is not None else {}),
        window=half, sign=sign,
        **({"per_record": per_record} if per_record is not None else {}),
        **({"seed": seed} if sign == "random" else {}),
        **({"skip_bos": skip_bos} if feats is None else {}),
        over=over,
        description=(
            (f"Corpus windows drawn at random under seed {seed}, read on " if sign == "random"
             else "The corpus windows whose token most excites ")
            + target
            + ", the " + ("drawn" if sign == "random" else "exciting")
            + " token marked by `hit` among the window's tokens."))


class Strip:
    def __init__(self, sides: tuple[str, ...]):
        self.kept: dict[str, list[tuple[float, dict[str, Any]]]] = {side: [] for side in sides}
        self.n_tokens = 0
        self.total = self.total_sq = 0.0
        self.lo, self.hi = float("inf"), float("-inf")

    def take(self, record: Mapping[str, Any], toks: Sequence[str], values: np.ndarray, *, first: int,
             half: int, k: int, cap: int, draw: np.ndarray | None, fires_only: bool) -> None:
        seen = values[first:]
        if not seen.size:
            return
        self.n_tokens += int(seen.size)
        self.total += float(seen.sum())
        self.total_sq += float(np.sum(seen.astype(np.float64) ** 2))
        self.lo, self.hi = min(self.lo, float(seen.min())), max(self.hi, float(seen.max()))

        def window_at(pos: int, value: float) -> dict[str, Any]:
            a, b = max(first, pos - half), min(len(toks), pos + half + 1)
            return {"id": f"{record.get('id')}:{pos}",
                    "coords": {**(record.get("coords") or {}), "record": record.get("id"),
                               "position": int(pos)},
                    "value": round(float(value), 5),
                    "token": toks[pos],
                    "text": "".join(toks[a:b]),
                    "tokens": list(toks[a:b]),
                    "values": [round(float(v), 5) for v in values[a:b]],
                    "hit": int(pos - a)}

        for side, kept in self.kept.items():
            if side == "random":
                at = np.arange(first, len(values))
                if fires_only:
                    at = at[values[at] != 0]
                for pos in at[np.argsort(draw[at], kind="stable")][:cap]:
                    kept.append((float(draw[pos]), window_at(int(pos), values[pos])))
                kept.sort(key=lambda t: t[0])
            elif side == "high":
                for pos in first + np.argsort(-seen)[:cap]:
                    kept.append((float(values[pos]), window_at(int(pos), values[pos])))
                kept.sort(key=lambda t: -t[0])
            else:
                for pos in first + np.argsort(seen)[:cap]:
                    kept.append((float(values[pos]), window_at(int(pos), values[pos])))
                kept.sort(key=lambda t: t[0])
            del kept[k:]

    def moments(self) -> dict[str, Any]:
        n = max(self.n_tokens, 1)
        mean = self.total / n
        var = max(self.total_sq / n - mean * mean, 0.0)
        return {"n_tokens": self.n_tokens, "mean": round(mean, 5), "sd": round(float(np.sqrt(var)), 5),
                "min": round(self.lo, 5), "max": round(self.hi, 5)}
