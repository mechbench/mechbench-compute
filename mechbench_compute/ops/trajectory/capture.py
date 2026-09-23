from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import lexicon
from mechbench_compute import points as hookpoints
from mechbench_compute import shapes as S
from mechbench_compute.lexicon._base import In, Op, Otherwise, Output, P

OP = Op(
    name="trajectory/capture",
    requires="mlx-local",
    summary=(
        "Follow the residual stream along an axis — one position through "
        "every layer, or one layer along every position of a text — and "
        "record the vector (or its coordinate along a direction) at each "
        "step."
    ),
    description="""\
One forward pass per record. With `axis: "layers"` the block reads the
vector at one `position` after each of `layers`; step *k* is the *k*-th
layer. With `axis: "positions"` it reads one `layer`'s vector at each of the
chosen `positions`; step *k* is the *k*-th position read, which for
`"generated"` is the *k*-th generated token.

A corpus-scale trajectory is large (200 stories × 160 steps × the model
width), so there are three ways to keep it an object:

* `max_steps` — stop after that many steps per record.
* `pool` — one vector per record, the reduction over a window of its
  steps (`{"reduce": "mean", "over": {"range": [5, 30]}}`): what an
  outcome axis is fit on.
* `project` — a direction: read the scalar coordinate along it at capture
  time and emit no vectors at all. The trace itself, as numbers.

Every item carries the record's `coords`, which is what
`trajectory/aggregate` groups on. A measurement a record carries as a
field (what `text/measure` writes) becomes a coordinate through
`records/rename` — `{"opening": "coords.opening"}` — before the capture.

**Replay.** A record generated at trace fidelity carries the exact token ids
it was generated as and where generation began. `replay: "auto"` uses those
when present and tokenizes the text otherwise; re-tokenizing a generated
story can shift a token boundary, and the trace is the ground truth of what
the model saw.
""",
    inputs=(
        In("records", "records/record",
           "The texts, each a record with a prompt (`user`, `prompt` or "
           "`text`) or a `trace`, and optionally `coords` and a `subject` "
           "when `position` is `\"subject\"`. A document collection is read "
           "the same way.", many=True),
        In("project", "direction/vector",
           "A direction: read each step's scalar coordinate along it at "
           "capture time and emit coordinates instead of vectors.",
           required=False),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('trajectory/point', collection=True, doc='One item per record per step: `{id, coords, space, step, position, token, norm, vector}`, with `vocab` (a distribution) when asked for, and `n_pooled` plus `steps` when reduced. With `project`, a collection of `activations/coordinate` instead: `coord` and the direction\'s identity in place of the vector, `projected: true` in the header. The header carries `axis`, `point`, `layers`, `position`/`positions`, `d_model` and `replay` (`"trace"`, `"text"` or `"mixed"`).',
                 otherwise=(Otherwise("activations/coordinate", collection=True, port="project"),)),
    params=(
        P("axis", "string",
          "`\"layers\"`: one position through every layer. `\"positions\"`: "
          "one layer along the sequence.",
          "layers", choices=("layers", "positions")),
        P("layers", "list[int] | \"all\"",
          "For `axis: \"layers\"`, the layers to step through.",
          "all"),
        P("layer", "int",
          "For `axis: \"positions\"`, the one layer to read along the "
          "sequence. (A one-element `layers` list is accepted too.)",
          None),
        P("position", "selector",
          "For `axis: \"layers\"`, which token to follow through the layers: "
          "`\"last\"`, `\"all\"`, a list of indices (negative from the end), "
          "`{\"tokens\": [...]}`, `{\"range\": [a, b]}`, `{\"after\": n}`, "
          "`\"subject\"` or `\"generated\"`, resolving to one position.",
          "last"),
        P("positions", "selector",
          "For `axis: \"positions\"`, which positions to step along: `\"last\"`, "
          "`\"all\"`, a list of indices (negative from the end), `{\"tokens\": "
          "[...]}`, `{\"range\": [a, b]}`, `{\"after\": n}`, `\"subject\"` or "
          "`\"generated\"` — `\"generated\"` is the story, not the prompt.",
          "generated"),
        P("max_steps", "int",
          "Stop after this many steps per record.",
          None),
        P("pool", "object",
          "Emit one vector per record — the reduction over a window of its "
          "steps — instead of one per step: `{\"reduce\": \"mean\" | \"max\", "
          "\"over\": <selector>}`, `over` counting steps from the trajectory's "
          "own start, so `{\"range\": [5, 30]}` is steps 5 … 29.",
          None, value="pool"),
        P("point", "string",
          "Which residual stream to read: `\"resid_post\"` (after each "
          "layer) or `\"resid_pre\"` (before it).",
          "resid_post", choices=("resid_post", "resid_pre"), value="point"),
        P("replay", "string",
          "`\"auto\"`: use the record's stored token ids when it has a "
          "trace, else tokenize its text. `\"trace\"`: require the trace. "
          "`\"text\"`: always tokenize the text.",
          "auto", choices=("auto", "trace", "text")),
        P("vocab_top", "int",
          "Also record each step's distribution through the unembedding "
          "(`vocab`, with this many top tokens), a lens reading per step. "
          "`0` records none.",
          0),
    ),
    example={
        "model": {"$param": "model"},
        "axis": "positions",
        "layer": 12,
        "positions": "generated",
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/stories"}}, "project": {"$ref": {"bench": "you/lab/outcome_axis"}}},
)


def run(ctx, inputs, params):
    """trajectory/capture — one position's vector at every layer, or
    one layer's vector at every
    position along a sequence, replayed from the trace when the
    records carry one."""

    model = ctx.model(params.get("model"))
    records = lexicon.items_of(inputs.get("records") or [])
    return capture(
        model, records, params, project=inputs.get("project"),
        on_item=ctx.on_item, on_start=ctx.on_start)


AXES = ("layers", "positions")


def _read_record_coords(record: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """The record's coordinates, as every item carries them. A document
    item keeps its coords under `metadata`; the retired `label` field is
    read as the `label` coordinate so older records group as they did.
    A measurement a record carries as a field becomes a coordinate
    through `records/rename` upstream, not here."""
    coords = dict(record.get("coords") or (record.get("metadata") or {}).get("coords") or {})
    label = record.get("label")
    if label is not None and "label" not in coords:
        coords["label"] = label
    return coords


def _trace_ids(record: Mapping[str, Any]) -> tuple[list[int] | None, int | None]:
    """(token_ids, generation_start) from a trace-fidelity item, or
    (None, None) when the record carries no trace."""
    trace = record.get("trace")
    if not isinstance(trace, Mapping):
        return None, None
    ids = trace.get("token_ids")
    if not isinstance(ids, list) or not ids:
        return None, None
    start = None
    spans = trace.get("generation_spans") or []
    if spans and isinstance(spans[0], Mapping):
        s = spans[0].get("token_start")
        start = int(s) if isinstance(s, int) else None
    return [int(t) for t in ids], start


def capture(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    project: Mapping[str, Any] | None = None,
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """The mechanism: one `trajectory/point` per (record, step), or the
    reduced or projected forms `params` asks for."""
    import mlx.core as mx

    from mechbench_compute import Capture
    from mechbench_compute import positions as POS
    from mechbench_compute.distill import render
    from mechbench_compute.interp import MAX_VECTOR_FLOATS, resolve_layers

    axis = str(params.get("axis", "layers"))
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}: one of {AXES}")
    point = hookpoints.residual(params.get("point"))
    replay = str(params.get("replay", "auto"))
    if replay not in ("auto", "trace", "text"):
        raise ValueError("replay must be 'auto', 'trace' or 'text'")
    vocab_top = int(params.get("vocab_top", 0) or 0)
    max_steps = params.get("max_steps")
    if not records:
        raise ValueError("trajectory/capture needs at least one record")
    # Two ways to keep a corpus-scale trajectory small enough to be an
    # object (200 stories × 160 steps × d_model is ~80M floats):
    #   reduce: "mean" — ONE pooled vector per record over the selected
    #           steps (a window like {"range": [5, 30]} via `steps`) —
    #           what an outcome axis is fit on.
    #   project: <direction> — read the scalar coordinate along a
    #           direction AT capture time and emit no vectors at all —
    #           the trace itself, 200 × 160 numbers.
    pool = POS.pool_spec(params)
    direction = project
    dvec = None
    if direction is not None:
        from mechbench_compute import directions as dirs

        if not isinstance(direction, Mapping) or "vector" not in direction:
            raise ValueError("`project` must be a direction record")
        dvec = dirs.coerce_array(direction)
        if dvec.shape[0] != model.arch.d_model:
            raise ValueError(
                f"project: direction has {dvec.shape[0]} dims; the model has "
                f"{model.arch.d_model}")

    if axis == "layers":
        layers = resolve_layers(params.get("layers"), model.arch.n_layers)
        position = params.get("position", "last")
        steps_per = len(layers)
    else:
        layer_spec = params.get("layer", params.get("layers"))
        if isinstance(layer_spec, list):
            if len(layer_spec) != 1:
                raise ValueError(
                    "a positions-axis trajectory reads ONE layer; pass `layer`")
            layer_spec = layer_spec[0]
        if layer_spec is None:
            raise ValueError("a positions-axis trajectory needs `layer`")
        layers = resolve_layers(int(layer_spec), model.arch.n_layers)
        position = params.get("positions", "generated")
        steps_per = int(max_steps) if max_steps else None

    width = model.arch.d_model
    # The cap counts the floats this node will EMIT, not the ones it
    # reads: `project` emits one scalar per step and no vectors at all,
    # and `reduce` emits one pooled vector per record. Counting steps ×
    # width regardless would refuse exactly the two configurations that
    # exist to stay under it.
    if direction is not None:
        per_record = 0
    elif pool:
        per_record = width
    else:
        per_record = None if steps_per is None else steps_per * width
    if per_record:
        total = len(records) * per_record
        if total > MAX_VECTOR_FLOATS:
            raise ValueError(
                f"{len(records)} records × {per_record // width} steps × "
                f"{width} dims = {total} floats exceeds the "
                f"{MAX_VECTOR_FLOATS} cap — set `pool`, `project`, or "
                "`max_steps`, or capture fewer records")
    if on_start:
        on_start(len(records))

    cap = Capture.residual(layers, point=hookpoints.side(point))
    tok = model.tokenizer
    mid = S.model_id_of(model)
    rows: list[dict[str, Any]] = []
    used_trace = 0
    for record in records:
        ids_list, gen_start = (None, None)
        if replay in ("auto", "trace"):
            ids_list, gen_start = _trace_ids(record)
        if ids_list is None:
            if replay == "trace":
                raise ValueError(
                    f"record {record.get('id')!r} has no trace and replay is "
                    "'trace' — score/generate at fidelity 'trace' first")
            r = render(model, record)
            ids = r.array
            prompt_len = r.prompt_len
        else:
            used_trace += 1
            ids = mx.array([ids_list])
            prompt_len = gen_start
        arr = np.array(ids).reshape(-1)
        seq_len = int(arr.shape[0])
        toks = [tok.decode([int(t)]) for t in arr]
        sel = dict(tokens=toks, record=record, prompt_len=prompt_len, gen_start=gen_start)
        result = model.run(ids, interventions=[cap])
        coords = _read_record_coords(record, params)

        def sp(layer: int) -> dict[str, Any]:
            return S.space(model=mid, layer=layer, point=point, d=width)

        if axis == "layers":
            pos = POS.one(position, seq_len, **sel)
            for step, layer in enumerate(layers):
                t = result.cache[f"blocks.{layer}.{point}"]
                v = t[0, pos, :].astype(mx.float32)
                mx.eval(v)
                rows.append(_project_row(
                    _build_point(record, coords, sp(layer), step, pos, np.array(v), tok, arr,
                                 model, vocab_top), np.array(v), dvec, direction))
        else:
            layer = layers[0]
            t = result.cache[f"blocks.{layer}.{point}"]
            seq = t[0].astype(mx.float32)
            mx.eval(seq)
            mat = np.array(seq)
            idx = POS.resolve(position, seq_len, **sel)
            if max_steps:
                idx = idx[: int(max_steps)]
            if not idx:
                raise ValueError(
                    f"record {record.get('id')!r}: no positions in the window")
            if pool:
                # `pool.over` selects among the trajectory's STEPS.
                over = [idx[s] for s in POS.resolve(pool["over"], len(idx))]
                v, n_pooled = POS.pooled(mat, over, pool["reduce"])
                row = _build_point(record, coords, sp(layer), 0, over[0] if over else idx[0], v, tok, arr, model, 0)
                row.pop("token", None)
                row.update({"n_pooled": n_pooled, "pool": pool})
                rows.append(_project_row(row, v, dvec, direction))
            else:
                if dvec is None:
                    running = len(rows) + len(idx)
                    if running * width > MAX_VECTOR_FLOATS:
                        raise ValueError(
                            f"trajectory exceeds the {MAX_VECTOR_FLOATS}-float "
                            f"cap at record {record.get('id')!r}; set `max_steps`, "
                            "`pool`, or `project`, or capture fewer records")
                for step, p in enumerate(idx):
                    row = _build_point(record, coords, sp(layer), step, p, mat[p], tok, arr,
                                       model, vocab_top)
                    rows.append(_project_row(row, mat[p], dvec, direction))
        if on_item:
            on_item()

    from mechbench_compute.lexicon import kinds as K

    # A projected trajectory is a collection of coordinates, not of
    # points without their vectors.
    return K.collection(
        "activations/coordinate" if dvec is not None else "trajectory/point", rows,
        axis=axis,
        point=point,
        layers=layers,
        position=str(position) if axis == "layers" else None,
        positions=(position if axis == "positions" else None),
        d_model=width,
        replay="trace" if used_trace == len(records)
               else ("text" if used_trace == 0 else "mixed"),
        n_items=len(records),
        projected=dvec is not None,
        **({"max_steps": int(max_steps)} if max_steps else {}),
        **({"pool": pool} if pool else {}),
    )


def _build_point(record, coords, sp, step, pos, vec: np.ndarray, tok, arr, model,
                 vocab_top: int) -> dict[str, Any]:
    """One `trajectory/point`: a vector item with its step and position."""
    row = S.vector(
        vec, sp, id=record.get("id"), coords=coords,
        token=S.token(tok, int(arr[pos])) if pos < len(arr) else None,
        step=int(step), position=int(pos))
    if vocab_top:
        row["vocab"] = _unembed_vector(model, vec, vocab_top)
    return row


def _project_row(row: dict[str, Any], vec: np.ndarray,
                 dvec: np.ndarray | None,
                 direction: Mapping[str, Any] | None) -> dict[str, Any]:
    """With a direction, a step is its scalar coordinate and carries no
    vector — the trace itself, small enough to be an object."""
    if dvec is None:
        return row
    return S.coordinate(
        float(np.asarray(vec, dtype=np.float32) @ dvec), row["space"], direction,
        id=row.get("id"), coords=row.get("coords"), token=row.get("token"),
        step=row["step"], position=row["position"],
        n_pooled=row.get("n_pooled"), steps=row.get("steps"))


def _unembed_vector(model, vec: np.ndarray, k: int) -> dict[str, Any]:
    """The vector through the unembedding: the lens reading of this
    point, as a distribution."""
    import mlx.core as mx

    logits = model.project_to_logits(mx.array(vec)[None, :]).astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp)
    return S.distribution(np.array(lp).reshape(-1), model.tokenizer, top_k=k)
