from __future__ import annotations

from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.distill import render
from mechbench_compute.interp.read_record_coords import read_record_coords
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="activations/capture-tokens",
    requires="mlx-local",
    summary=(
        "Capture the residual-stream vector at EVERY position of each "
        "prompt — one vector per token, each carrying that token's own "
        "surprisal."
    ),
    description="""\
`activations/capture` reads ONE position per record: a decision point, a
subject, a span pooled into a single vector. Some questions are about the
sequence itself — which way the residual moves as a token's surprisal
rises, how a representation builds across a passage — and those need a
row per token.

Each vector carries its token's surprisal in bits, on the `surprisal`
coordinate, because the forward pass that produced the vector already
computed it. Joining the two afterwards by (record, position) would be
awkward and a chance to misalign them by one — the off-by-one that makes
a probe fit the NEXT token's difficulty. The first position of a sequence
has no predecessor and so carries no surprisal, rather than a zero, which
would read as a confident prediction of the first token.

A per-token capture is a different order of magnitude from one vector per
record. As JSON it has a ceiling — a stored object may not exceed 64 MiB —
and `storage: "json"` refuses beyond it, naming the levers: fewer layers,
a narrower `positions` (a chat template's own tokens are rarely the
question), a larger `every`, or fewer records. `storage: "tensor"` has no
such ceiling: the rows are written as safetensors shards beside the
object (`<label>/shards/…`), the object itself is the header with the
shards named, and a reader takes the rows one shard at a time — a
hundred thousand tokens at a layer is an ordinary capture. `"auto"`, the
default, is json under the ceiling and tensor above it.

Fitted against the surprisal it carries, by `direction/regress`, this is
the surprise-direction probe.
""",
    inputs=(
        In("records", "records/record", "The prompts to read.", many=True),
        In("adapter", "adapter/lora", "An adapter to fuse first.", required=False),
    ),
    output=Output('activations/vector', collection=True, doc='One item per record per kept position per layer, with `position` and `surprisal` among its coords and the token as `token`. The header carries `model`, `point`, `source`, `position` (the selector), `every`, `layers` and `d_model`; under tensor storage also `storage: "tensor"`, `shards` (name, rows, size, sha256 each), `n_items` and `d`, with `items` empty — the rows are the shards.'),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to run over.",
          "all"),
        P("point", "string",
          "Which residual stream to read: `\"resid_post\"` (after each "
          "layer) or `\"resid_pre\"` (before it).",
          "resid_post", choices=("resid_post", "resid_pre"), value="point"),
        P("positions", "selector",
          "Which positions to read: `\"last\"`, `\"all\"`, a list of indices "
          "(negative from the end), `{\"tokens\": [...]}`, `{\"range\": [a, "
          "b]}`, `{\"after\": n}`, `\"subject\"` or `\"generated\"`. Unlike "
          "`capture`'s single position, every position named is read.",
          "all"),
        P("every", "int",
          "Take one position in n of those named — a way under the ceiling "
          "that keeps the whole sequence's span rather than its first part.",
          1),
        P("storage", "string",
          "`\"json\"`: the rows in the object, under the ceiling. "
          "`\"tensor\"`: the rows as shards beside it, no ceiling. "
          "`\"auto\"`: json under the ceiling, tensor above.",
          "auto", choices=("auto", "json", "tensor")),
    ),
    example={
        "model": {"$param": "model"},
        "layers": [21],
        "positions": {"after": 20},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/passages"}}},
)


def run(ctx, inputs, params):
    """activations/capture-tokens — one vector per token, each
    carrying that token's own surprisal."""

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return capture_tokens(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


#: A per-token capture materialises one vector per (record, position,
#: layer) — a different order of magnitude from one vector per record,
#: so it has its own ceiling.
#:
#: The number is set by what the platform can STORE, not by taste. A
#: canonical float costs about 8.3 bytes on the wire, and the API
#: refuses an object over 64 MiB, so ~8.1M floats is the wall. This
#: ceiling must stay under it: one set above it lets the whole capture
#: run and fails at the emit, which is the worst place to learn a limit.
MAX_TOKEN_VECTOR_FLOATS = 7_000_000


def capture_tokens(
    model,
    records,
    params,
    *,
    on_start=None,
    on_item=None,
):
    """The residual at EVERY position, one vector per token.

    `capture` reads one position per record — a decision point, a
    subject, a pooled span. Some questions are about the sequence
    itself: which way the residual moves as a token's surprisal rises,
    how a representation builds across a passage. Those need a row per
    token, which is this.

    Each vector carries the token's own surprisal, in bits, because the
    forward pass that produced the vector already computed it. Joining
    the two afterwards, by (record, position), would be both awkward and
    a chance to misalign them by one — the off-by-one that makes a
    surprisal probe fit the NEXT token's difficulty.
    """
    layers = resolve_layers(params.get("layers", "all"), model.arch.n_layers)
    point = hookpoints.normalize(str(params.get("point", "resid_post")))
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
    # Where the rows go: `"json"` is the collection itself, under the cap
    # a stored object can hold; `"tensor"` writes the rows to shards
    # beside the object, with no cap but disk; `"auto"` is json under the
    # cap and tensor above it.
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

    cap = Capture.residual(layers, point=hookpoints.side(point))
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
        coords = read_record_coords(record, params)
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
    return load_kinds().collection("activations/vector", rows, **header)
