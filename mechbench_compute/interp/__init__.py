"""Interpretability primitives as protocol operations (the
mechbench-experiments port).

The original step-XX scripts each hand-rolled a loop around the same
three moves: run the model with an intervention, read something out of
the residual stream, compare. The intervention layer (interventions.py)
already made those moves declarative; this module makes them PROTOCOL
BLOCKS, so the experiments become graphs anyone can run, re-run, and
diff on the platform:

- ``ablate_layers``     — steps 02/04/34/35 and the legacy flat kind:
                          per-layer (or per-sublayer) Δ log p sweeps.
- ``residual_vectors``  — steps 01/08/10/11/12's shared substrate:
                          residual-stream vectors at (layer, position)
                          per condition, as data other blocks consume.
- ``residual_divergence`` — the matched-pair mechanism (000050/052):
                          run a pair of prompts, cosine-compare the
                          residual streams per (layer, position).
- ``vector_similarity`` — steps 10/11/28's readout (pure, no model):
                          cosine matrix + separation metrics over
                          labeled vectors. Lives in blocks.PURE_BLOCKS.

Every op renders its records one way (`distill.render`): a condition
(`user`, optional `system` and `prefill`) through the model's chat
template, a bare `text`/`prompt` record raw. So an ablation sweep can
read at a decision point inside an assistant turn, exactly where
`logits/read` reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from mechbench_compute import points as P
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute.distill import encode, render
from mechbench_compute.interventions import Ablate, Capture
from mechbench_compute.interp.k import _K  # noqa: F401
from mechbench_compute.interp.last_logp import _last_logp  # noqa: F401
from mechbench_compute.interp.pair import _pair  # noqa: F401
from mechbench_compute.interp.render_text import _render_text  # noqa: F401
from mechbench_compute.interp.resolve_layers import _resolve_layers  # noqa: F401
from mechbench_compute.interp.target_of import _target_of  # noqa: F401
from mechbench_compute.interp.target_token_id import _target_token_id  # noqa: F401
from mechbench_compute.interp.tracked_ids import _tracked_ids  # noqa: F401

#: Refuse vector payloads past this many floats — a mistyped layer list
#: must not emit a gigabyte of CBOR. ~16 MB of float64 at the cap.
MAX_VECTOR_FLOATS = 2_000_000

#: The points `intervene/ablate-layers` can zero at a layer, by the point's
#: name. On a non-MatFormer model `gate_out` has no hook and the run
#: refuses with the arch's own error.
_ABLATE_AT: dict[str, Callable[[int], Any]] = {
    "attn_out": Ablate.attention,
    "mlp_out": Ablate.mlp,
    "gate_out": Ablate.side_channel,
}


def _ablation_points(spec: Any) -> list[str]:
    """The `point` param of `intervene/ablate-layers`: one name or a list of
    them, each a sub-layer output; the default is both, the whole
    layer's contribution."""
    if spec is None:
        return ["attn_out", "mlp_out"]
    names = [spec] if isinstance(spec, str) else [str(p) for p in spec]
    out = [P.normalize(n) for n in names]
    bad = [n for n in out if n not in _ABLATE_AT]
    if bad or not out:
        raise ValueError(
            f"intervene/ablate-layers zeroes a sub-layer output — one or more of "
            f"{sorted(_ABLATE_AT)} — not {bad or spec!r}")
    return out


def _own_top1_if_different(model, tok: int, lp: np.ndarray | None) -> dict[str, Any]:
    """`{"own_top1": token}` when the model's own top-1 under `lp` is not
    the target being measured — empty when it is, or when there is no
    baseline to ask (task 000597).

    A tracked target that is not the model's answer is often the point
    (measure THIS token's dependence), so this refuses nothing. But it
    is also how a mis-tokenized target hides: `tracked` says to include
    the leading space, which is right for a raw completion and wrong
    after a chat template's assistant prefix, where `" Paris"` and
    `"Paris"` are different tokens. A reader who sees the model's own
    answer beside the target knows at a glance which case they are in.
    """
    if lp is None:
        return {}
    top = int(np.argmax(lp))
    if top == tok:
        return {}
    return {"own_top1": {**S.token(model.tokenizer, top),
                         "logp": round(float(lp[top]), 4)}}


def _coords_of(record: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """The record's coordinates. A grouping is a coordinate; the retired
    `label` field is read as the `label` coordinate so older records
    group as they did. (A document keeps its coords under `metadata`.)"""
    coords = dict(record.get("coords") or (record.get("metadata") or {}).get("coords") or {})
    label = record.get("label")
    if label is not None and "label" not in coords:
        coords["label"] = label
    return coords


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
    points = _ablation_points(params.get("point"))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("ablate/layers needs at least one condition")
    if on_start:
        on_start(len(records) * (len(layers) + 1))

    def intervene(layer: int) -> list[Any]:
        # Zeroing both sub-layer outputs leaves the stream as it entered
        # the layer — the whole-layer skip, on the path that has always
        # computed it.
        if set(points) == {"attn_out", "mlp_out"}:
            return [Ablate.layer(layer)]
        return [_ABLATE_AT[p](layer) for p in points]
    rows: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    damage_by_layer: dict[int, list[float]] = {i: [] for i in layers}
    for record in records:
        r = render(model, record)
        ids = r.array
        base_lp = _last_logp(model.run(ids).logits)
        if on_item:
            on_item()
        tok, _ = _target_of(model, record, params, base_lp)
        baseline = float(base_lp[tok])
        for layer in layers:
            lp = _last_logp(model.run(ids, interventions=intervene(layer)).logits)
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
            # model completing raw text answers with function words. The
            # target above was the only place that showed, as a symptom;
            # this says it (task 000596).
            "template": "chat" if r.chat else "raw",
            **_own_top1_if_different(model, tok, base_lp),
        })

    return _K().collection(
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


def residual_vectors(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Residual-stream vectors at (layers × position) per condition —
    the substrate every geometry experiment reads. Labels ride along
    (`label` field, or the coords key named by params.label_coord) so
    similarity blocks can group without re-parsing ids."""
    point = P.residual(params.get("point"))
    source = str(params.get("source", "resid"))
    if source not in ("resid", "queries", "keys"):
        raise ValueError(
            f"unknown source {source!r}: 'resid', 'queries' or 'keys'")
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    position = params.get("position", "last")
    pool = POS.pool_spec(params)
    if not records:
        raise ValueError("residuals/vectors needs at least one condition")
    # A document with no text cannot be embedded. Dropping one changes
    # n, so it happens only when asked for and the ids are reported.
    skipped: list[str] = []
    if bool(params.get("skip_empty", False)):
        def _has_text(r):
            v = r.get("user") or r.get("prompt") or r.get("text")
            return isinstance(v, str) and bool(v.strip())

        skipped = [str(r.get("id")) for r in records if not _has_text(r)]
        records = [r for r in records if _has_text(r)]
        if not records:
            raise ValueError(
                "residuals/vectors: every record was empty under skip_empty")
    # Q/K live in per-head subspaces (step 28's question: which heads
    # specialize?), so those sources emit one row per (layer, head).
    n_heads = (model.arch.n_heads if source == "queries"
               else model.arch.n_kv_heads if source == "keys" else 1)
    width = (model.arch.d_model if source == "resid"
             else model.arch.d_model // model.arch.n_heads)
    total = len(records) * len(layers) * n_heads * width
    if total > MAX_VECTOR_FLOATS:
        raise ValueError(
            f"{len(records)} conditions × {len(layers)} layers × "
            f"{n_heads} heads × {width} dims = {total} floats exceeds "
            f"the {MAX_VECTOR_FLOATS} cap — capture fewer layers or "
            "conditions"
        )
    if on_start:
        on_start(len(records))

    if source == "resid":
        cap = Capture.residual(layers, point=P.side(point))
    elif source == "queries":
        cap = Capture.queries(layers)
    else:
        cap = Capture.keys(layers)
    rows: list[dict[str, Any]] = []
    mid = S.model_id_of(model)
    for record in records:
        r = render(model, record)
        ids = r.array
        toks = r.tokens(model.tokenizer)
        result = model.run(ids, interventions=[cap])
        # Pooling reads the selected positions, so no single position is
        # resolved — and a record whose `position` would not resolve (no
        # `subject`, say) is still poolable.
        sel = dict(tokens=toks, record=record, prompt_len=r.prompt_len)
        pos = None if pool else POS.one(position, len(r.ids), **sel)
        over = POS.resolve(pool["over"], len(r.ids), **sel) if pool else None
        coords = _coords_of(record, params)
        read_token = None if pool else S.token(model.tokenizer, r.ids[pos])
        for layer in layers:
            if source == "resid":
                t = result.cache[f"blocks.{layer}.{point}"]
                if pool:
                    seq = t[0].astype(mx.float32)
                    mx.eval(seq)
                    v, n_pooled = POS.pooled(np.array(seq), over, pool["reduce"])
                else:
                    v = t[0, pos, :].astype(mx.float32)
                    mx.eval(v)
                    v, n_pooled = np.array(v), None
                rows.append(S.vector(
                    v, S.space(model=mid, layer=layer, point=point, d=width),
                    id=record.get("id"), coords=coords, token=read_token,
                    n_pooled=n_pooled))
            else:
                key = ("q" if source == "queries" else "k")
                t = result.cache[f"blocks.{layer}.attn.{key}"]
                arr = np.array(t.astype(mx.float32))[0]  # [heads, L, hd]
                for head in range(arr.shape[0]):
                    if pool:
                        v, n_pooled = POS.pooled(arr[head], over, pool["reduce"])
                    else:
                        v, n_pooled = arr[head, pos, :], None
                    rows.append(S.vector(
                        v, S.space(model=mid, layer=layer, point=f"attn.{key}",
                                   d=width, head=head),
                        id=record.get("id"), coords=coords, token=read_token,
                        n_pooled=n_pooled))
        if on_item:
            on_item()
    return _K().collection(
        "activations/vector", rows,
        model=mid,
        point=point,
        source=source,
        # A pooled record says so where a reader looks for the
        # position, rather than reporting a position it never read.
        position="pooled" if pool else str(position),
        **({"pool": pool} if pool else {}),
        layers=layers,
        d_model=width,
        **({"skipped_empty": skipped} if skipped else {}),
    )


#: A per-token capture materialises one vector per (record, position,
#: layer) — a different order of magnitude from one vector per record,
#: so it has its own ceiling.
#:
#: The number is set by what the platform can STORE, not by taste. A
#: canonical float costs about 8.3 bytes on the wire, and the API
#: refuses an object over 64 MiB, so ~8.1M floats is the real wall.
#: 0.107.x set this at 20M: the op then passed its own check, did the
#: whole capture, and died at the emit — the worst place to learn a
#: limit. A ceiling that does not bind is not a ceiling.
MAX_TOKEN_VECTOR_FLOATS = 7_000_000


def capture_tokens(
    model,
    records,
    params,
    *,
    on_start=None,
    on_item=None,
):
    """The residual at EVERY position, one vector per token (000594).

    `capture` reads one position per record — a decision point, a
    subject, a pooled span. Some questions are about the sequence
    itself: which way the residual moves as a token's surprisal rises,
    how a representation builds across a passage. Those need a row per
    token, and there was no way to ask for one.

    Each vector carries the token's own surprisal, in bits, because the
    forward pass that produced the vector already computed it. Joining
    the two afterwards, by (record, position), would be both awkward and
    a chance to misalign them by one — the off-by-one that makes a
    surprisal probe fit the NEXT token's difficulty.
    """
    layers = _resolve_layers(params.get("layers", "all"), model.arch.n_layers)
    point = P.normalize(str(params.get("point", "resid_post")))
    positions = params.get("positions", "all")
    every = max(1, int(params.get("every", 1)))
    width = model.arch.d_model

    kept_per_record = []
    for record in records:
        r = render(model, record)
        sel = dict(tokens=r.tokens(model.tokenizer), record=record,
                   prompt_len=r.prompt_len)
        idx = POS.resolve(positions, len(r.ids), **sel)[::every]
        kept_per_record.append((record, r, idx))

    total = sum(len(idx) for _, _, idx in kept_per_record) * len(layers) * width
    # Where the rows go (000613): `"json"` is the collection as it always
    # was, under the cap a stored object can hold; `"tensor"` writes the
    # rows to shards beside the object, with no cap but disk; `"auto"`
    # is json under the cap and tensor above it.
    storage = str(params.get("storage", "auto"))
    if storage not in ("auto", "json", "tensor"):
        raise ValueError(f"storage is 'auto', 'json' or 'tensor', not {storage!r}")
    if storage == "auto":
        storage = "tensor" if total > MAX_TOKEN_VECTOR_FLOATS else "json"
    if storage == "json" and total > MAX_TOKEN_VECTOR_FLOATS:
        raise ValueError(
            f"{len(records)} records × {sum(len(i) for _, _, i in kept_per_record)} "
            f"kept positions × {len(layers)} layers × {width} dims = {total} "
            f"floats exceeds the {MAX_TOKEN_VECTOR_FLOATS} cap (a result "
            "object may not exceed 64 MiB, and a float costs ~8.3 bytes "
            "stored) — set `storage: \"tensor\"` to write the rows as shards, "
            "capture fewer layers, narrow `positions` (the chat template's "
            "own tokens are rarely the question), raise `every` to take one "
            "position in n, or capture fewer records")
    if on_start:
        on_start(len(records))

    cap = Capture.residual(layers, point=P.side(point))
    mid = S.model_id_of(model)
    rows: list[dict[str, Any]] = []
    writer = None
    if storage == "tensor":
        import tempfile

        from mechbench_compute import tensors

        writer = tensors.ShardWriter(tempfile.mkdtemp(prefix="mechbench-tensor-"))
    for record, r, idx in kept_per_record:
        ids = r.array
        result = model.run(ids, interventions=[cap])
        # Surprisal of token i given everything before it, from the SAME
        # forward pass that produced the vectors — the logits are already
        # in hand, and a second pass would be both slower and a chance
        # for the two to disagree. Position 0 has no predecessor and so
        # carries none: a zero there would read as a confident
        # prediction of the first token.
        seq = list(r.ids)
        lg = result.logits[0, :-1, :].astype(mx.float32)
        tgt = mx.array(seq[1:])
        lp = (mx.take_along_axis(lg, tgt[:, None], axis=-1)[:, 0]
              - mx.logsumexp(lg, axis=-1))
        mx.eval(lp)
        surp = -np.array(lp) / np.log(2.0)
        coords = _coords_of(record, params)
        for pos in idx:
            token = S.token(model.tokenizer, r.ids[pos])
            bits = None if pos == 0 else round(float(surp[pos - 1]), 4)
            for layer in layers:
                t = result.cache[f"blocks.{layer}.{point}"]
                v = t[0, pos, :].astype(mx.float32)
                mx.eval(v)
                row = S.vector(
                    np.array(v),
                    S.space(model=mid, layer=layer, point=point, d=width),
                    id=record.get("id"),
                    coords={**coords, "position": int(pos),
                            **({"surprisal": bits} if bits is not None else {})},
                    token=token)
                if writer is not None:
                    writer.add(row)
                else:
                    rows.append(row)
        if on_item:
            on_item()
    header = dict(model=mid, point=point, source="resid",
                  position=str(positions), every=every, layers=layers, d_model=width)
    if writer is not None:
        from mechbench_compute import tensors

        return tensors.collection("activations/vector", writer.close(), **header)
    return _K().collection("activations/vector", rows, **header)


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
        point = P.normalize(str(params.get("point", "mlp.act")))
        vec = None
    else:
        sp = dirs.space_of(direction)
        layer = params.get("layer", sp.get("layer"))
        if layer is None:
            raise ValueError("the direction names no layer; give `layer`")
        layer = int(layer)
        point = P.normalize(str(params.get("point") or sp.get("point") or "resid_post"))
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


def residual_divergence(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Matched-pair divergence: run prompts `a` and `b`, and per
    (layer, position) report 1 − cosine of the residual streams. The
    map shows WHERE a one-token change ripples (000050/000052).

    Pairs must tokenize to equal lengths — that is what 'matched'
    means; unequal pairs are reported as errors, not silently aligned.
    """
    point = P.residual(params.get("point"))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("residuals/divergence needs at least one pair")
    if on_start:
        on_start(len(records) * 2)

    cap = Capture.residual(layers, point=P.side(point))
    pairs: list[dict[str, Any]] = []
    for record in records:
        a, b = _pair(record)
        ids_a = _render_text(model, record, a)
        ids_b = _render_text(model, record, b)
        len_a = int(np.array(ids_a).shape[-1])
        len_b = int(np.array(ids_b).shape[-1])
        if len_a != len_b:
            pairs.append(S.grid(
                record.get("id"), ["layer", "position"], {},
                coords=record.get("coords"),
                error=(f"prompts tokenize to different lengths ({len_a} vs "
                       f"{len_b}) — a matched pair must match")))
            if on_item:
                on_item()
                on_item()
            continue
        run_a = model.run(ids_a, interventions=[cap])
        if on_item:
            on_item()
        run_b = model.run(ids_b, interventions=[cap])
        if on_item:
            on_item()
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids_a).reshape(-1)]
        matrix: list[list[float]] = []
        for layer in layers:
            va = np.array(run_a.cache[f"blocks.{layer}.{point}"][0]
                          .astype(mx.float32))
            vb = np.array(run_b.cache[f"blocks.{layer}.{point}"][0]
                          .astype(mx.float32))
            na = np.linalg.norm(va, axis=-1)
            nb = np.linalg.norm(vb, axis=-1)
            cos = (va * vb).sum(axis=-1) / np.maximum(na * nb, 1e-9)
            matrix.append([round(float(1.0 - c), 5) for c in cos])
        pairs.append(S.grid(
            record.get("id"), ["layer", "position"], {"divergence": matrix},
            tokens=tokens, coords=record.get("coords")))
    return _K().collection(
        "activations/divergence", pairs,
        point=point,
        layers=layers,
        description=(
            "1 − cosine similarity of the two residual streams per "
            "(layer, position). 0 = identical; the map shows where a "
            "one-token change ripples."
        ),
    )


def lens_positions(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Step 08 as a block: project every layer's residual through the
    unembedding at every position and follow one target token — where
    in the sequence, and at what depth, does the answer become
    visible? Rank 0 means the target is that position's top readout."""
    from mechbench_compute import lens

    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("lens/positions needs at least one condition")
    if on_start:
        on_start(len(records))

    cap = Capture.residual(layers, point="post")
    rows: list[dict[str, Any]] = []
    for record in records:
        ids = render(model, record).array
        result = model.run(ids, interventions=[cap])
        tok, _ = _target_of(model, record, params, _last_logp(result.logits))
        ranks, logprobs = lens.logit_lens_per_position(
            model, result.cache, tok, layers=layers)
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids).reshape(-1)]
        rows.append(S.grid(
            record.get("id"), ["layer", "position"],
            {"logprob": [[round(float(x), 4) for x in r] for r in logprobs],
             "rank": [[int(x) for x in r] for r in ranks]},
            tokens=tokens, coords=record.get("coords"),
            target=S.token(model.tokenizer, tok)))
        if on_item:
            on_item()
    return _K().collection(
        "logits/lens", rows,
        layers=layers,
        description=(
            "Logit-lens readout of the target token at every (layer, "
            "position): log p and rank of the target when each layer's "
            "residual is projected straight through the unembedding."
        ),
    )


#: Attention matrices are quadratic in sequence length; refuse a
#: capture that would emit more than this many floats.
MAX_ATTN_FLOATS = 2_000_000


def attention_patterns(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Steps 05/06: post-softmax attention weights per head at chosen
    layers. Layers must be named explicitly — every layer of every
    head of a long prompt is a picture nobody asked for."""
    spec = params.get("layers")
    if spec in (None, "all"):
        raise ValueError(
            "attention/patterns needs an explicit layers list — "
            "attention weights are per-head and quadratic in prompt "
            "length, so name the layers you want to look at"
        )
    layers = _resolve_layers(spec, model.arch.n_layers)
    if not records:
        raise ValueError("attention/patterns needs at least one condition")
    if on_start:
        on_start(len(records))

    cap = Capture.attn_weights(layers)
    rows: list[dict[str, Any]] = []
    total_floats = 0
    for record in records:
        ids = render(model, record).array
        result = model.run(ids, interventions=[cap])
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids).reshape(-1)]
        seq = len(tokens)
        total_floats += len(layers) * model.arch.n_heads * seq * seq
        if total_floats > MAX_ATTN_FLOATS:
            raise ValueError(
                f"attention capture would exceed {MAX_ATTN_FLOATS} floats "
                "— fewer layers, shorter prompts, or fewer conditions"
            )
        weight = []
        for layer in layers:
            w = result.cache[f"blocks.{layer}.attn.weights"]
            arr = np.array(w.astype(mx.float32))[0]  # [heads, L, S]
            weight.append([[[round(float(x), 4) for x in r] for r in h] for h in arr])
        rows.append(S.grid(
            record.get("id"), ["layer", "head", "query", "key"], {"weight": weight},
            tokens=tokens, coords=record.get("coords")))
        if on_item:
            on_item()
    return _K().collection(
        "activations/attention", rows,
        n_heads=model.arch.n_heads,
        layers=layers,
        description=(
            "Post-softmax attention weights per head: row = the "
            "attending position, column = the attended-to position."
        ),
    )


def ablate_heads(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Step 07: zero one head at a time across (layers × heads) and
    measure Δ log p of the target — the head-level version of the
    layer sweep. Progress ticks per (condition, layer)."""
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
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
        base_lp = _last_logp(model.run(ids).logits)
        if on_item:
            on_item()
        tok, _ = _target_of(model, record, params, base_lp)
        baseline = float(base_lp[tok])
        metas.append({
            "id": record.get("id"),
            "target": S.token(model.tokenizer, tok),
            "baseline_logp": round(baseline, 4),
            "template": "chat" if r.chat else "raw",
            **_own_top1_if_different(model, tok, base_lp),
        })
        for li, layer in enumerate(layers):
            for head in range(n_heads):
                lp = _last_logp(model.run(
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


def logit_attribution(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Steps 32/33 as a block: direct logit attribution per layer.

    One forward per condition captures every layer's residual plus the
    final norm's scale; the residual stream is decomposed into exactly
    additive components (embedding, then each layer's delta), and each
    component's contribution to the target logit is read through the
    norm-folded unembed (apply_ln, task 000142).

    SELF-VALIDATING: every row reports its additivity residual — the
    summed contributions minus the model's true final logit. A reader
    never has to take the decomposition on faith.
    """
    from mechbench_compute import attribution

    apply_ln = bool(params.get("apply_ln", True))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
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

    per_head_layers = _resolve_layers(
        params.get("per_head_layers"), model.arch.n_layers
    ) if params.get("per_head_layers") else []
    interventions = [
        Cap.residual(layers, point="post"),
        Cap.residual([0], point="pre"),
        Cap.final_norm_scale(),
    ]
    if per_head_layers:
        # Forces the manual attention path at these layers — per-head
        # writes (000145) cost real time, so they are opt-in by layer.
        interventions.append(Cap.per_head_out(per_head_layers))
    rows: list[dict[str, Any]] = []
    for record in records:
        r = render(model, record)
        ids = r.array
        result = model.run(ids, interventions=interventions)
        lp = _last_logp(result.logits)
        tok, tracked = _target_of(model, record, params, lp)
        # Two tracked tokens: the contributions are to the DIFFERENCE of
        # their logits (target minus the second). A record from before
        # the spellings were one may still name the second as `contrast`.
        others = [t for t in tracked.values() if t != tok]
        contrast = record.get("contrast")
        ctok = (_target_token_id(model, str(contrast)) if contrast
                else (others[0] if others else None))

        acc = attribution.accumulated_resid(result.cache, include_pre=True)
        components = np.diff(acc, axis=0, prepend=np.zeros_like(acc[:1]))
        # components[0] = embedding stream, components[i] = layer i-1's delta
        ln_scale = np.array(
            mx.array(result.cache["final_norm.scale"]).astype(mx.float32)
        ).reshape(-1)
        targets = [tok] if ctok is None else [tok, ctok]
        attrs = attribution.logit_attrs(
            model, components, targets,
            apply_ln=apply_ln, ln_scale=ln_scale)
        contrib = attrs[:, 0] if ctok is None else attrs[:, 0] - attrs[:, 1]

        # The honesty number: does the decomposition sum to the truth?
        # The comparison lives in PRE-softcap space — the decomposition
        # is linear and the cap is not, so a capped "true" logit would
        # disagree structurally (gemma4 caps; e2b's first run showed it).
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
            **_own_top1_if_different(model, tok, lp),
            per_head=per_head or None,
            additivity={
                "summed": round(summed, 3),
                "true_logit": round(true_logit, 3),
                "residual": round(summed - true_logit, 3),
            }))
        if on_item:
            on_item()
    return _K().collection(
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


def steer_inject(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    inputs: Mapping[str, Any] | None = None,
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Steering as a block (epic 000131, arc B): build a direction from
    a labeled residual_vectors record (centroid of `positive` minus
    centroid of `negative` at the injection layer) and ADD it to each
    eval prompt's residual stream at (layer, position), sweeping alpha.
    The readout is the final-position top-k under each alpha — alpha 0
    is the built-in control.

    The direction comes from DATA flowing through the graph, not from a
    hardcoded vector: the same residuals/vectors block that measures
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
    if not isinstance(vectors, Mapping) or _K().item_kind_of(vectors) != "activations/vector":
        raise ValueError(
            "intervene/steer needs a collection of activations/vector on "
            "its `vectors` port — the same block that measures geometry "
            "arms the intervention")
    rows_at = [r for r in _K().items_of(vectors) if S.layer_of(r) == layer]
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
        tracked = _tracked_ids(model, record, tracked=params.get("tracked"))
        for alpha in alphas:
            interventions = (
                [] if alpha == 0.0
                else [Patch.add(layer, pos_idx, value, alpha=alpha)]
            )
            lp = _last_logp(model.run(ids, interventions=interventions).logits)
            out_rows.append({
                "id": record.get("id"),
                "coords": dict(record.get("coords") or {}),
                "factor": alpha,
                **S.distribution(lp, model.tokenizer, top_k=top_k, tracked=tracked),
            })
            if on_item:
                on_item()
    return _K().collection(
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


