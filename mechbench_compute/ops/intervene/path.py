from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.interp import read_last_logp, read_pair, render_text, resolve_target
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="intervene/path",
    requires="mlx-local",
    summary=(
        "A sender's effect through ONE receiver — its output corrupted, "
        "everything between the two held clean — which is how a circuit's "
        "edges are established rather than its nodes."
    ),
    description="""\
`intervene/patch` changes what every downstream component sees, so a head
that "matters" may matter only because the thing it feeds matters. This
asks the narrower question a circuit claim needs: does what THIS head
writes reach THAT head's query, as opposed to reaching the answer by some
other route (Wang et al. 2022; Goldowsky-Dill et al. 2023)?

Per sender: the clean prompt runs with the sender's output replaced by
its value on the corrupt prompt and **every component between the sender
and the receiver frozen at its clean value**, so the only thing that
changed at the receiver's input is what arrived along the path; the
receiver's new output is then written into an otherwise clean run, where
the metric is read. `delta` is the change from the clean baseline.

A **sender** writes into the residual stream: a head's own contribution
(`attn.per_head_out`, read before `o_proj` concatenates it, which is what
makes one head separable from its neighbours), a whole layer's
`attn_out`, or its `mlp_out`. `senders: "all-heads"` sweeps every head of
every earlier layer; `"all-layers"` sweeps the two branches of each.

A **receiver** reads: a head's `attn.q`, `attn.k` or `attn.v` — "which
heads feed this one's query" — or `logits`, which asks what reaches the
answer directly rather than through anything else. A receiver of `logits`
needs one pass per sender instead of two.

The pair is a record's `a` (clean) and `b` (corrupt), as
`intervene/patch` takes them, and they must tokenize to the same length.
Cost is two forward passes per sender per record (one for `logits`), so
`all-heads` on a 35-layer model is a few hundred passes: find the
candidates with `intervene/patch method: "attribution"` first, then
establish the edges here.

**Read the small numbers as zero.** A path effect is a narrow channel by
construction, and most senders reach a given receiver not at all: on a
factual pair through Gemma 4 E2B the median sender moves the answer's
logit by exactly 0 and one moves it by 1.5. Below about a quarter of a
logit the model's own bf16 arithmetic is the larger term, so use `logit`
(the most nearly linear metric), a pair whose clean and corrupt answers
are far apart, and treat a delta of that size as no path at all.
""",
    inputs=(
        In("records", "records/pair",
           "Pairs, each with prompt strings `a` (clean) and `b` (corrupt), "
           "and optionally its own `tracked`.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('intervene/readout', collection=True,
                  doc='One item per record per sender: `delta` (the change in the target\'s metric from the clean baseline), `value`, `cell` (the sender named), and `coords` carrying the sender\'s `layer`, `point` and `head` beside the receiver\'s `into_layer`, `into_point`, `into_head`. The header carries `metric`, `receiver`, `n_senders` and `target`.'),
    params=(
        P("receiver", "object",
          "What reads: `{\"point\": \"attn.q\", \"layer\": 23, \"head\": 5}`, "
          "or `{\"point\": \"logits\"}` for what reaches the answer directly.",
          {"point": "logits"}, fields=(
              P("point", "string", "Where it reads.", "logits",
                choices=("attn.q", "attn.k", "attn.v", "logits")),
              P("layer", "int", "Its layer; none for `logits`.", None),
              P("head", "int", "Its head, at a point that has them.", None),
          )),
        P("senders", "json",
          "`\"all-heads\"`, `\"all-layers\"`, one object `{point, layer, "
          "head}`, or a list of them. Every sender must be earlier than the "
          "receiver.",
          "all-heads"),
        P("metric", "string",
          "What the delta is measured in: the target's log-probability, its "
          "probability, or its raw logit.",
          "logprob", choices=("logprob", "prob", "logit")),
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is "
          "the target — the answer whose path is traced; the clean prompt's "
          "top‑1 by default. Each is tokenized as a continuation of the rendered "
          "prompt: with a leading space after a raw prompt, without one after a "
          "chat template's assistant prefix. A record's own `tracked` takes "
          "precedence; with none named, the model's own top-1 prediction for "
          "that prompt is the target, and a target that differs from it is "
          "reported beside it.",
          None),
    ),
    example={"model": {"$param": "model"},
             "receiver": {"point": "attn.q", "layer": 23, "head": 5},
             "senders": "all-heads", "metric": "logit"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/pairs"}}},
)


def run(ctx, inputs, params):
    """intervene/path (task 000607): a sender's effect through one
    receiver, with everything between them held at its clean value —
    the edge test a circuit claim stands on."""

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return run_path_patch(model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


#: What a sender may be: a component that WRITES into the residual
#: stream. A head's own contribution is read before `o_proj` concatenates
#: it, which is what makes one head separable from its neighbours.
SENDER_POINTS = ("attn.per_head_out", "attn_out", "mlp_out")


#: What a receiver may be: a component's INPUT — a head's query, key or
#: value — or the model's own output.
RECEIVER_POINTS = ("attn.q", "attn.k", "attn.v", "logits")


class SpecError(ValueError):
    """A path that cannot be run, named."""


def _name(point: str, layer: int | None) -> str:
    return point if layer is None else f"blocks.{layer}.{point}"


def _make_head_writer(value: mx.array, head: int | None) -> Callable:
    """A hook that writes `value` — at one head, when the point has a
    head axis and a head was named."""
    def fn(act: mx.array, info) -> mx.array:
        if head is None:
            return value.astype(act.dtype)
        mask = np.zeros(act.shape[1], dtype=bool)
        mask[head] = True
        view = [1] * len(act.shape)
        view[1] = act.shape[1]
        return mx.where(mx.array(mask).reshape(view), value.astype(act.dtype), act)
    return fn


def _parse_end(spec: Any, *, what: str, points: Sequence[str],
               n_layers: int, default_point: str) -> dict[str, Any]:
    if spec is None:
        spec = {}
    if not isinstance(spec, Mapping):
        raise SpecError(f"`{what}` is an object: {{point, layer, head}}")
    point = str(spec.get("point", default_point))
    if point not in points:
        raise SpecError(
            f"`{what}.point` is one of {', '.join(points)} — not {point!r}")
    layer = spec.get("layer")
    if point == "logits":
        layer = None
    elif layer is None:
        raise SpecError(f"`{what}` names no layer")
    else:
        layer = int(layer)
        if not 0 <= layer < n_layers:
            raise SpecError(f"`{what}.layer` {layer} is outside this model "
                            f"({n_layers} layers)")
    head = spec.get("head")
    if head is not None and point in ("attn_out", "mlp_out", "logits"):
        raise SpecError(f"`{what}.point` {point!r} has no heads; drop `head`")
    return {"point": point, "layer": layer,
            "head": None if head is None else int(head)}


def _collect_senders(params: Mapping[str, Any], receiver: Mapping[str, Any],
                     n_layers: int, n_heads: int) -> list[dict[str, Any]]:
    """The senders to score. `all-heads` and `all-layers` sweep every
    component that could reach the receiver — which is every one in a
    strictly earlier layer, since a residual stream carries only what
    has already been written."""
    # One word: `senders` takes one object as readily as a list.
    spec = params.get("senders")
    last = n_layers if receiver["layer"] is None else receiver["layer"]
    if isinstance(spec, str):
        if spec == "all-heads":
            return [{"point": "attn.per_head_out", "layer": layer, "head": head}
                    for layer in range(last) for head in range(n_heads)]
        if spec == "all-layers":
            return [{"point": point, "layer": layer, "head": None}
                    for layer in range(last) for point in ("attn_out", "mlp_out")]
        raise SpecError(
            f"`senders` is 'all-heads', 'all-layers', one object or a list — "
            f"not {spec!r}")
    given = spec if isinstance(spec, (list, tuple)) else [spec]
    out = [_parse_end(one, what="sender", points=SENDER_POINTS,
                      n_layers=n_layers, default_point="attn.per_head_out")
           for one in given]
    late = [s for s in out if s["layer"] is not None and s["layer"] >= last]
    if late:
        raise SpecError(
            f"a sender must be earlier than its receiver (layer {last}); "
            f"layer {late[0]['layer']} is not — a residual stream carries only "
            "what has already been written")
    return out


def _freeze_off_path(cache: Mapping[str, Any], sender: Mapping[str, Any],
                     receiver: Mapping[str, Any], n_layers: int) -> dict[str, Callable]:
    """Hooks that hold every component NOT on the path at its clean
    value: everything between the sender and the receiver, and the MLP
    of the sender's own layer when the sender is its attention. What is
    left changing at the receiver is what arrived along the path."""
    ls = int(sender["layer"])
    last = n_layers if receiver["layer"] is None else int(receiver["layer"])
    hooks: dict[str, Callable] = {}
    if sender["point"] != "mlp_out":
        name = _name("mlp_out", ls)
        hooks[name] = _make_head_writer(cache[name], None)
    for layer in range(ls + 1, last):
        for point in ("attn_out", "mlp_out"):
            name = _name(point, layer)
            hooks[name] = _make_head_writer(cache[name], None)
    return hooks


def run_path_patch(model, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any],
        on_item: Callable | None = None,
        on_start: Callable[[int], None] | None = None) -> dict[str, Any]:
    """`intervene/path`: what each sender contributes to the receiver,
    and through it to the answer."""
    from mechbench_compute.lexicon import kinds as K

    n_layers, n_heads = model.arch.n_layers, model.arch.n_heads
    receiver = _parse_end(params.get("receiver"), what="receiver",
                          points=RECEIVER_POINTS, n_layers=n_layers,
                          default_point="logits")
    senders = _collect_senders(params, receiver, n_layers, n_heads)
    if not senders:
        raise SpecError("no sender to score")
    metric = str(params.get("metric", "logprob"))
    if metric not in ("logprob", "prob", "logit"):
        raise SpecError(f"unknown metric {metric!r}: 'logprob', 'prob' or 'logit'")
    if not records:
        raise SpecError("path patching needs at least one pair")
    if on_start:
        on_start(len(records) * len(senders))

    recv_name = None if receiver["point"] == "logits" else _name(
        receiver["point"], receiver["layer"])
    # Every component's output, for the freezing; plus the receiver's.
    clean_names = [_name(p, layer) for layer in range(n_layers)
                   for p in ("attn_out", "mlp_out")]
    rows: list[dict[str, Any]] = []
    for record in records:
        clean, corrupt = read_pair(record)
        ids_clean = render_text(model, record, clean)
        ids_corrupt = render_text(model, record, corrupt)
        n_clean = int(np.array(ids_clean).shape[-1])
        n_corrupt = int(np.array(ids_corrupt).shape[-1])
        if n_clean != n_corrupt:
            raise SpecError(
                f"pair {record.get('id')!r} tokenizes to different lengths "
                f"({n_clean} vs {n_corrupt}) — positions cannot align under patching")
        want = clean_names + ([recv_name] if recv_name else [])
        a = model.run(ids_clean, interventions=[Capture.at(want)])
        clean_lp = read_last_logp(a.logits)
        tok, _ = resolve_target(model, record, params, clean_lp)

        def read(logits, tok: int = tok) -> float:
            row = logits[0, -1, :].astype(mx.float32)
            mx.eval(row)
            arr = np.array(row)
            if metric == "logit":
                return float(arr[tok])
            lp = arr - float(mx.logsumexp(mx.array(arr)))
            return float(np.exp(lp[tok])) if metric == "prob" else float(lp[tok])

        baseline = read(a.logits)
        sender_names = sorted({_name(s["point"], s["layer"]) for s in senders})
        b = model.run(ids_corrupt, interventions=[Capture.at(sender_names)])

        for sender in senders:
            s_name = _name(sender["point"], sender["layer"])
            hooks = _freeze_off_path(a.cache, sender, receiver, n_layers)
            hooks[s_name] = _make_head_writer(b.cache[s_name], sender["head"])
            if recv_name is None:
                value = read(model.run(ids_clean, hooks=hooks).logits)
            else:
                c = model.run(ids_clean, hooks=hooks,
                              interventions=[Capture.at([recv_name])])
                d = model.run(ids_clean,
                              hooks={recv_name: _make_head_writer(c.cache[recv_name],
                                                                  receiver["head"])})
                value = read(d.logits)
            rows.append({
                "id": record.get("id"),
                "coords": {**(record.get("coords") or {}),
                           "layer": sender["layer"], "point": sender["point"],
                           **({"head": sender["head"]} if sender["head"] is not None else {}),
                           "into_layer": receiver["layer"],
                           "into_point": receiver["point"],
                           **({"into_head": receiver["head"]}
                              if receiver["head"] is not None else {})},
                "factor": 1.0,
                "cell": (f"L{sender['layer']}"
                         + (f"H{sender['head']}" if sender["head"] is not None else "")
                         + f":{sender['point']}"),
                "delta": round(value - baseline, 5),
                "value": round(value, 5),
            })
            if on_item:
                on_item(f"{record.get('id')}:{rows[-1]['cell']}", rows[-1])
    return K.collection(
        "intervene/readout", rows,
        metric=metric,
        receiver=receiver,
        n_senders=len(senders),
        target=S.token(model.tokenizer, tok),
        description=(
            "Path patching: the change in the target's "
            f"{metric} when each sender's output is corrupted and only the "
            "path to the receiver carries it — everything between the two "
            "held at its clean value."),
    )
