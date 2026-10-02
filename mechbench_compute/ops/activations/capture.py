from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute.arrays import read_f32
from mechbench_compute.attribution import decompose_attn_out
from mechbench_compute.distill import render
from mechbench_compute.interp.check_written import check_written
from mechbench_compute.interp.constants import MAX_VECTOR_FLOATS
from mechbench_compute.interp.read_record_coords import read_record_coords
from mechbench_compute.interp.load_kinds import load_kinds
from mechbench_compute.interp.resolve_layers import resolve_layers
from mechbench_compute.interventions import Capture
from mechbench_compute.lexicon._base import In, Op, Output, P, Resume
from mechbench_compute.spans import add_to_span

POINTS = ("resid_post", "resid_pre", "attn_out", "mlp_out", "gate_out", "attn.weights")

WRITES = ("attn_out", "mlp_out", "gate_out")

WEIGHTS = "attn.weights"

OP = Op(
    name="activations/capture",
    needs=frozenset({"model.forward"}),
    resume=Resume("restart"),
    summary=(
        "Capture the residual-stream vector of each prompt, or what a layer "
        "writes into it, at chosen layers and a chosen position (or pooled "
        "over the sequence) — the raw material for every geometry measurement."
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

`point` says what is read there. `resid_post` and `resid_pre` are the
residual stream leaving and entering each layer. `attn_out`, `mlp_out`
and `gate_out` are what the layer writes into it: the attention block's
output after any norm the architecture applies to it, the MLP block's,
and the per-layer input gate's on a checkpoint that has one (Gemma 4's E
models). A checkpoint whose layers write no gate refuses `gate_out` by
name, with the code `POINT_ABSENT`. The architecture's residual law
relates them: `resid_post` is `resid_pre` plus the writes (on Gemma 4,
times the layer's scalar), so the writes at one layer say which of them
moved a dimension of the stream.

`heads` splits `attn_out` by head, one vector per head: the head's output
through its slice of `o_proj`, then through the norm the architecture
applies after `o_proj` (Gemma 3 and Gemma 4), whose divisor, the root
mean square of the whole `o_proj` output, is held at its value in the
run. A layer's heads sum to its `attn_out`; an `o_proj` bias, on a
checkpoint that has one, belongs to no head. Each vector's `space.head`
says whose it is.

`point: "attn.weights"` reads the attention weights of the heads `heads`
names: at each query position `position` names (any number of them, each
read apart), the weights over the key positions, which sum to 1. Each
item carries its query position as `coords.position` and the query token
as `token`; pooled, it is the reduction of the weights over the pooled
query positions.

Split heads, attention weights and queries or keys are read from
attention computed head by head at the captured layers, which in bf16 is
not bit-identical to the fused attention the other reads run: the header
then says `attention_path: "per_head"`, and a comparison with a read on
the fused path has that difference as its floor.

`top: k` adds to each vector its `k` largest coordinates by size, as
`top: [{dim, value, share}]`, the largest first, `share` being the
coordinate's part of the squared norm, and `rms_without_top`, the root
mean square of the vector with those coordinates set to zero (the whole
vector's is `norm / √d`): which dimensions carry a vector's size,
without a loop over its width. On attention weights `top` is the `k` key
positions with the most weight, `[{position, token, weight}]`.

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
           "document collection is read the same way."
           " A model reads each record as [its kind](/kinds/records/record/) says: `user` in the chat template, `prompt` and `text` raw.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('activations/vector', collection=True, doc='One item per record per layer (per head, for Q/K sources and split heads; per head and query position, for attention weights): `{id, coords, space, vector, norm}`, plus `token` (the token read, when not pooled; on attention weights the query token, its position in `coords.position`), `n_pooled` when pooled, and with `top` set, `top` and `rms_without_top` (on attention weights, `top` alone). The header carries `model`, `point`, `source`, `position` (`"pooled"` when pooled), `layers`, `d_model` (absent on attention weights, whose width is the record\'s length), `heads` and `attention_path: "per_head"` when heads were read, `top` when set, and `skipped_empty` listing any records dropped under `skip_empty`.'),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to run over.",
          "all"),
        P("point", "string",
          "What to read: `\"resid_post\"` (the residual stream after each "
          "layer), `\"resid_pre\"` (before it), a layer's write into it "
          "(`\"attn_out\"`, `\"mlp_out\"`, or `\"gate_out\"`, the per-layer "
          "input gate, which a checkpoint without one refuses), or "
          "`\"attn.weights\"`, the attention weights of the heads `heads` names.",
          "resid_post", choices=POINTS, value="point"),
        P("heads", "list[int] | \"all\"",
          "With `attn_out`, split the attention write by these heads, one "
          "vector each, which sum to `attn_out`; with `attn.weights`, the "
          "heads whose weights are read, which must be named. `\"all\"` is "
          "every head.",
          None),
        P("source", "string",
          "What to capture: `\"resid\"` (what `point` names, one vector per "
          "layer), `\"queries\"` or `\"keys\"` (attention Q or K, one vector "
          "per layer per head).",
          "resid", choices=("resid", "queries", "keys")),
        P("position", "selector",
          "Which token's vector to read: `\"last\"`, `\"all\"`, a list of "
          "indices (negative from the end), `{\"tokens\": [...]}`, `{\"range\": "
          "[a, b]}`, `{\"after\": n}`, `\"subject\"` or `\"generated\"`, "
          "resolving to a single position; on `attn.weights`, the query "
          "positions, any number, each read apart. Ignored when `pool` is set.",
          "last"),
        P("pool", "object",
          "Read a set of positions and reduce them to one vector: "
          "`{\"reduce\": \"mean\" | \"max\", \"over\": <selector>}` — "
          "`{\"reduce\": \"mean\", \"over\": {\"range\": [5, 30]}}` is the "
          "mean over positions 5 … 29 (a document's body after its "
          "envelope); `\"over\": \"all\"` is the whole sequence.",
          None, value="pool"),
        P("top", "int",
          "Add each vector's `top` largest coordinates by size, with each "
          "one's share of the squared norm, and the vector's root mean square "
          "without them; on attention weights, the `top` key positions with "
          "the most weight and their tokens. 0 adds nothing.",
          0),
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
    source = str(params.get("source", "resid"))
    if source not in ("resid", "queries", "keys"):
        raise ValueError(
            f"unknown source {source!r}: 'resid', 'queries' or 'keys'")
    point = read_point(model, params.get("point"), source)
    heads = read_heads(params.get("heads"), point, model.arch.n_heads)
    top = read_top_count(params.get("top", 0))
    layers = resolve_layers(params.get("layers"), model.arch.n_layers)
    position = params.get("position", "last")
    pool = POS.pool_spec(params)
    if not records:
        raise ValueError("residuals/vectors needs at least one condition")
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
    n_heads = (model.arch.n_heads if source == "queries"
               else model.arch.n_kv_heads if source == "keys" else len(heads or [None]))
    width = (model.arch.d_model if source == "resid"
             else model.arch.d_model // model.arch.n_heads)
    total = len(records) * len(layers) * n_heads * width
    if point != WEIGHTS and total > MAX_VECTOR_FLOATS:
        raise ValueError(
            f"{len(records)} conditions × {len(layers)} layers × "
            f"{n_heads} heads × {width} dims = {total} floats exceeds "
            f"the {MAX_VECTOR_FLOATS} cap — capture fewer layers or "
            "conditions"
        )
    if on_start:
        on_start(len(records))

    cap = build_capture(source, point, heads, layers)
    rows: list[dict[str, Any]] = []
    mid = S.model_id_of(model)
    n_weights = 0
    for record in records:
        r = render(model, record)
        ids = r.array
        toks = r.tokens(model.tokenizer)
        sel = dict(tokens=toks, record=record, prompt_len=r.prompt_len)
        over = POS.resolve(pool["over"], len(r.ids), **sel) if pool else None
        queries = [] if pool or point != WEIGHTS else POS.resolve(position, len(r.ids), **sel)
        if point == WEIGHTS:
            n_weights += len(layers) * len(heads) * max(1, len(queries)) * len(r.ids)
            if n_weights > MAX_VECTOR_FLOATS:
                raise ValueError(
                    f"attention weights of {len(heads)} heads × {len(layers)} layers "
                    f"over records of {len(r.ids)} tokens exceed the "
                    f"{MAX_VECTOR_FLOATS} cap — fewer heads, layers, query "
                    "positions or records")
        add_to_span(tokens_in=int(ids.size))
        result = model.run(ids, interventions=[cap])
        coords = read_record_coords(record, params)
        if point == WEIGHTS:
            rows += read_weight_rows(model, result.cache, record, r.ids, coords, mid=mid,
                                     layers=layers, heads=heads, queries=queries,
                                     over=over, pool=pool, top=top)
            if on_item:
                on_item()
            continue
        pos = None if pool else POS.one(position, len(r.ids), **sel)
        read_token = None if pool else S.token(model.tokenizer, r.ids[pos])
        for layer in layers:
            if heads is not None:
                split, _ = decompose_attn_out(model, result.cache, layer)
                for head in heads:
                    v, n_pooled = ((split[head, pos], None) if not pool
                                   else POS.pooled(split[head], over, pool["reduce"]))
                    sp = S.space(model=mid, layer=layer, point=point, d=width, head=head)
                    rows.append(build_row(v, sp, record, coords, read_token, n_pooled, top))
            elif source == "resid":
                t = result.cache[f"blocks.{layer}.{point}"]
                v, n_pooled = ((read_f32(t[0, pos, :]), None) if not pool
                               else POS.pooled(read_f32(t[0]), over, pool["reduce"]))
                rows.append(build_row(v, S.space(model=mid, layer=layer, point=point, d=width),
                                      record, coords, read_token, n_pooled, top))
            else:
                key = ("q" if source == "queries" else "k")
                arr = read_f32(result.cache[f"blocks.{layer}.attn.{key}"])[0]
                for head in range(arr.shape[0]):
                    if pool:
                        v, n_pooled = POS.pooled(arr[head], over, pool["reduce"])
                    else:
                        v, n_pooled = arr[head, pos, :], None
                    sp = S.space(model=mid, layer=layer, point=f"attn.{key}", d=width, head=head)
                    rows.append(build_row(v, sp, record, coords, read_token, n_pooled, top))
        if on_item:
            on_item()
    return load_kinds().collection(
        "activations/vector", rows,
        model=mid,
        point=point,
        source=source,
        position="pooled" if pool else str(position),
        **({"pool": pool} if pool else {}),
        layers=layers,
        d_model=None if point == WEIGHTS else width,
        **({"heads": heads} if heads is not None else {}),
        **({"attention_path": "per_head"} if heads is not None or source != "resid" else {}),
        **({"top": top} if top else {}),
        **({"skipped_empty": skipped} if skipped else {}),
    )


def read_point(model, spec: Any, source: str) -> str:
    point = hookpoints.normalize(spec)
    if point not in POINTS:
        raise ValueError(
            f"activations/capture reads {', '.join(POINTS)}, not {point!r}")
    if source != "resid" and point not in hookpoints.RESIDUAL:
        raise ValueError(
            f"source {source!r} reads attention's {source}; `point` {point!r} "
            "is read with source 'resid'")
    if point in WRITES:
        check_written(model, point)
    return point


def read_heads(spec: Any, point: str, n_heads: int) -> list[int] | None:
    if spec is None:
        if point == WEIGHTS:
            raise ValueError(
                f"`{WEIGHTS}` reads the heads `heads` names: a list of head "
                "indices, or \"all\"")
        return None
    if point not in ("attn_out", WEIGHTS):
        raise ValueError(
            f"`heads` splits `attn_out` by head or names the heads `{WEIGHTS}` "
            f"reads; `{point}` has no heads")
    if spec == "all":
        return list(range(n_heads))
    if not isinstance(spec, list) or not spec or not all(
            isinstance(h, int) and not isinstance(h, bool) and 0 <= h < n_heads for h in spec):
        raise ValueError(
            f"`heads` is \"all\" or a list of head indices from 0 to {n_heads - 1}, "
            f"not {spec!r}")
    if len(set(spec)) != len(spec):
        raise ValueError(f"`heads` names a head twice: {spec!r}")
    return list(spec)


def read_top_count(spec: Any) -> int:
    if spec is None:
        return 0
    if isinstance(spec, bool) or not isinstance(spec, int) or spec < 0:
        raise ValueError(f"`top` is a count of coordinates, 0 or more, not {spec!r}")
    return spec


def build_capture(source: str, point: str, heads: list[int] | None, layers: list[int]):
    if source == "queries":
        return Capture.queries(layers)
    if source == "keys":
        return Capture.keys(layers)
    if point == WEIGHTS:
        return Capture.attn_weights(layers)
    if heads is not None:
        return Capture.per_head_out(layers)
    if point in hookpoints.RESIDUAL:
        return Capture.residual(layers, point=hookpoints.side(point))
    return Capture.at(f"blocks.{i}.{point}" for i in layers)


def read_weight_rows(model, cache, record: Mapping[str, Any], ids: Sequence[int],
                     coords: Mapping[str, Any], *, mid: str | None, layers: list[int],
                     heads: list[int], queries: Sequence[int], over: Sequence[int] | None,
                     pool: Mapping[str, Any] | None, top: int) -> list[dict[str, Any]]:
    def read_key(j: int) -> dict[str, Any]:
        return S.token(model.tokenizer, ids[j])

    rows: list[dict[str, Any]] = []
    for layer in layers:
        w = read_f32(cache[f"blocks.{layer}.{WEIGHTS}"])[0]
        for head in heads:
            sp = S.space(model=mid, layer=layer, point=WEIGHTS, d=len(ids), head=head)
            if pool:
                v, n_pooled = POS.pooled(w[head], over, pool["reduce"])
                rows.append(build_row(v, sp, record, coords, None, n_pooled, top, read_key))
            for q in queries:
                rows.append(build_row(w[head, q], sp, record, {**coords, "position": int(q)},
                                      read_key(q), None, top, read_key))
    return rows


def build_row(v: Any, sp: Mapping[str, Any], record: Mapping[str, Any],
              coords: Mapping[str, Any], token: Mapping[str, Any] | None,
              n_pooled: int | None, top: int,
              read_key: Callable[[int], dict[str, Any]] | None = None) -> dict[str, Any]:
    item = S.vector(v, sp, id=record.get("id"), coords=coords, token=token, n_pooled=n_pooled)
    if top:
        x = np.asarray(v, dtype=np.float32).astype(np.float64)
        item.update(read_top(x, top, item["vector"]) if read_key is None
                    else read_top_keys(x, top, read_key, item["vector"]))
    return item


def read_top(x: np.ndarray, k: int, stored: Sequence[float]) -> dict[str, Any]:
    picked = rank_largest(np.abs(x), k)
    total = float(x @ x)
    rest = x.copy()
    rest[picked] = 0.0
    return {
        "top": [{"dim": int(i), "value": stored[i],
                 "share": round(float(x[i] * x[i]) / total, 6) if total else 0.0}
                for i in picked],
        "rms_without_top": round(float(np.sqrt(np.mean(rest * rest))), 6),
    }


def read_top_keys(x: np.ndarray, k: int, read_key: Callable[[int], dict[str, Any]],
                  stored: Sequence[float]) -> dict[str, Any]:
    return {"top": [{"position": int(j), "token": read_key(int(j)), "weight": stored[j]}
                    for j in rank_largest(x, k) if x[j] > 0]}


def rank_largest(size: np.ndarray, k: int) -> np.ndarray:
    k = min(k, size.size)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    cut = np.partition(size, size.size - k)[size.size - k]
    above = np.flatnonzero(size > cut)
    picked = np.concatenate([above, np.flatnonzero(size == cut)[:k - above.size]])
    return picked[np.lexsort((picked, -size[picked]))]
