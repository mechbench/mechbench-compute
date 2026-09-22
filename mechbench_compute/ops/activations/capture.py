from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute._mlx import mx
from mechbench_compute.distill import render
from mechbench_compute.interp.constants import MAX_VECTOR_FLOATS
from mechbench_compute.interp.read_record_coords import read_record_coords
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="activations/capture",
    requires="mlx-local",
    summary=(
        "Capture the residual-stream vector of each prompt at chosen layers "
        "and a chosen position (or pooled over the sequence) — the raw "
        "material for every geometry measurement."
    ),
    description="""\
One forward pass per record; for each requested layer the block records one
vector. Where in the sequence the vector is read is the important choice:

* `position: "last"` — the last token: right for a prompt whose meaning
  sits at its end (a question awaiting its answer).
* `position: "subject"` — the last token of the record's `subject` string
  within the prompt; a list of one index names that token.
* `pool` — read a set of positions and reduce them: `{"reduce": "mean",
  "over": "all"}` is the whole sequence, `{"reduce": "mean", "over":
  {"range": [5, 30]}}` a window. Right for a document: embedding a story
  at its last token embeds its *ending*, and a corpus with varied endings
  and identical middles would look varied.

Every item carries its `space` (`{model, layer, point, head?, d}`) and the
record's `coords`, which is what `geometry/compare`, `geometry/span` and
`direction/*` group on (their `axis` names the coordinate).

With `source: "queries"` or `"keys"` the block captures attention Q or K
vectors instead of the residual, one row per (layer, head).

The block refuses a capture of more than two million values; use fewer
layers or records.
""",
    inputs=(
        In("records", "records/record",
           "The prompts (`user`, `prompt` or `text`), each optionally with "
           "`coords`, and a `subject` when `position` is `\"subject\"`. A "
           "document collection is read the same way.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('activations/vector', collection=True, doc='One item per record per layer (per head, for Q/K sources): `{id, coords, space, vector, norm}`, plus `token` (the token read, when not pooled) and `n_pooled` when pooled. The header carries `model`, `point`, `source`, `position` (`"pooled"` when pooled), `layers`, `d_model`, and `skipped_empty` listing any records dropped under `skip_empty`.'),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to run over.",
          "all"),
        P("point", "string",
          "Which residual stream to read: `\"resid_post\"` (after each "
          "layer) or `\"resid_pre\"` (before it).",
          "resid_post", choices=("resid_post", "resid_pre"), value="point"),
        P("source", "string",
          "What to capture: `\"resid\"` (the residual stream, one vector per "
          "layer), `\"queries\"` or `\"keys\"` (attention Q or K, one vector "
          "per layer per head).",
          "resid", choices=("resid", "queries", "keys")),
        P("position", "selector",
          "Which token's vector to read: `\"last\"`, `\"all\"`, a list of "
          "indices (negative from the end), `{\"tokens\": [...]}`, `{\"range\": "
          "[a, b]}`, `{\"after\": n}`, `\"subject\"` or `\"generated\"`, "
          "resolving to a single position. Ignored when `pool` is set.",
          "last"),
        P("pool", "object",
          "Read a set of positions and reduce them to one vector: "
          "`{\"reduce\": \"mean\" | \"max\", \"over\": <selector>}` — "
          "`{\"reduce\": \"mean\", \"over\": {\"range\": [5, 30]}}` is the "
          "mean over positions 5 … 29 (a document's body after its "
          "envelope); `\"over\": \"all\"` is the whole sequence.",
          None, value="pool"),
        P("skip_empty", "bool",
          "Drop records with no text instead of refusing. The dropped ids "
          "are reported in `skipped_empty`, because dropping changes n.",
          False),
    ),
    example={
        "model": {"$param": "model"},
        "layers": [8, 12, 16],
        "pool": {"reduce": "mean", "over": {"range": [5, 30]}},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/stories"}}},
)


def run(ctx, inputs, params):
    """activations/capture — residual vectors at
    (layers × position) per condition, as data downstream blocks
    (vectors/similarity, future probes) consume."""

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return capture_residual_vectors(
        model, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


def capture_residual_vectors(
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
    point = hookpoints.residual(params.get("point"))
    source = str(params.get("source", "resid"))
    if source not in ("resid", "queries", "keys"):
        raise ValueError(
            f"unknown source {source!r}: 'resid', 'queries' or 'keys'")
    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
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
        cap = Capture.residual(layers, point=hookpoints.side(point))
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
        coords = read_record_coords(record, params)
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
    return load_kinds().collection(
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
