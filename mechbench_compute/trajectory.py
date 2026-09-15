"""Residual trajectories as objects (task 000368, lexicon epic 000364).

The lens computes a position's vector at every layer implicitly and
throws it away; experiment 014 computed a layer's vector at every
position along a story by hand. Both are the same object read along a
different axis, and this module makes it a kind:

    axis "layers"     one POSITION's vector at every captured layer —
                      the funnel (013), where in depth a commitment forms.
    axis "positions"  one LAYER's vector at every position along a
                      sequence — the trace (014), where in a story it
                      forms.

A trajectory is a collection of `trajectory/point` — vector items
(`space`, `vector`, `norm`) with `step`, `position` and the token read —
`step` indexing the axis. A projected one is a collection of
`activations/coordinate`. Four blocks:

    trajectory/capture    the model block: capture one per record.
    trajectory/project    scalar coordinate of every row along a
                          direction (directions.py), for a trace to read.
    trajectory/compare    two trajectories → per-step cosine, angle,
                          norm ratio, and the divergence step.
    trajectory/aggregate  group rows and reduce: mean trajectory + spread
                          per step; or a windowed mean per group emitted
                          as `residual_vectors`, so `direction/from-vectors`
                          reads it unchanged (014's outcome axis is
                          "lighthouse-story mean minus other-story mean
                          over tokens 5..30" — exactly that).

REPLAY. A stored corpus at trace fidelity carries the exact token ids a
story was generated as (`trace.token_ids`) and where generation began
(`trace.generation_spans`). `replay: "auto"` (the default) reads those
when present and tokenizes the text otherwise — re-tokenizing a
generated story can shift a boundary, and the trace is the ground
truth of what the model actually saw.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S

# --- shapes ------------------------------------------------------------------

AXES = ("layers", "positions")


def _coords_of(record: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
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


# --- capture (the model block) -----------------------------------------------


def capture(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    project: Mapping[str, Any] | None = None,
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """trajectory/capture."""
    import mlx.core as mx

    from mechbench_compute import Capture
    from mechbench_compute import positions as POS
    from mechbench_compute.distill import render
    from mechbench_compute.interp import MAX_VECTOR_FLOATS, _resolve_layers

    axis = str(params.get("axis", "layers"))
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}: one of {AXES}")
    point = P.residual(params.get("point"))
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
        dvec = dirs.as_array(direction)
        if dvec.shape[0] != model.arch.d_model:
            raise ValueError(
                f"project: direction has {dvec.shape[0]} dims; the model has "
                f"{model.arch.d_model}")

    if axis == "layers":
        layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
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
        layers = _resolve_layers(int(layer_spec), model.arch.n_layers)
        position = params.get("positions", "generated")
        steps_per = int(max_steps) if max_steps else None

    width = model.arch.d_model
    # The cap counts the floats this node will EMIT, not the ones it
    # reads: `project` emits one scalar per step and no vectors at all,
    # and `reduce` emits one pooled vector per record. Counting steps ×
    # width regardless refused exactly the two configurations that exist
    # to stay under it (a 100-story trace is 23M floats as vectors and
    # 15k numbers as coordinates).
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

    cap = Capture.residual(layers, point=P.side(point))
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
        coords = _coords_of(record, params)

        def sp(layer: int) -> dict[str, Any]:
            return S.space(model=mid, layer=layer, point=point, d=width)

        if axis == "layers":
            pos = POS.one(position, seq_len, **sel)
            for step, layer in enumerate(layers):
                t = result.cache[f"blocks.{layer}.{point}"]
                v = t[0, pos, :].astype(mx.float32)
                mx.eval(v)
                rows.append(_projected(
                    _row(record, coords, sp(layer), step, pos, np.array(v), tok, arr,
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
                row = _row(record, coords, sp(layer), 0, over[0] if over else idx[0], v, tok, arr, model, 0)
                row.pop("token", None)
                row.update({"n_pooled": n_pooled, "pool": pool})
                rows.append(_projected(row, v, dvec, direction))
            else:
                if dvec is None:
                    running = len(rows) + len(idx)
                    if running * width > MAX_VECTOR_FLOATS:
                        raise ValueError(
                            f"trajectory exceeds the {MAX_VECTOR_FLOATS}-float "
                            f"cap at record {record.get('id')!r}; set `max_steps`, "
                            "`pool`, or `project`, or capture fewer records")
                for step, p in enumerate(idx):
                    row = _row(record, coords, sp(layer), step, p, mat[p], tok, arr,
                               model, vocab_top)
                    rows.append(_projected(row, mat[p], dvec, direction))
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


def _row(record, coords, sp, step, pos, vec: np.ndarray, tok, arr, model,
         vocab_top: int) -> dict[str, Any]:
    """One `trajectory/point`: a vector item with its step and position."""
    row = S.vector(
        vec, sp, id=record.get("id"), coords=coords,
        token=S.token(tok, int(arr[pos])) if pos < len(arr) else None,
        step=int(step), position=int(pos))
    if vocab_top:
        row["vocab"] = _vocab(model, vec, vocab_top)
    return row


def _projected(row: dict[str, Any], vec: np.ndarray,
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


def _vocab(model, vec: np.ndarray, k: int) -> dict[str, Any]:
    """The vector through the unembedding: the lens reading of this
    point, as a distribution."""
    import mlx.core as mx

    logits = model.project_to_logits(mx.array(vec)[None, :]).astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp)
    return S.distribution(np.array(lp).reshape(-1), model.tokenizer, top_k=k)


# --- pure blocks over trajectories -------------------------------------------


def _trajectory_of(x: Any, port: str = "trajectory",
                   coords_ok: bool = False) -> Mapping[str, Any]:
    from mechbench_compute.lexicon import kinds as K

    if isinstance(x, Mapping):
        ik = K.item_kind_of(x)
        if ik == "trajectory/point" or (coords_ok and ik == "activations/coordinate"):
            return x
    raise ValueError(f"port {port!r} is not a collection of trajectory/point"
                     + (" or activations/coordinate" if coords_ok else ""))


def _rows(traj: Mapping[str, Any]) -> list[Any]:
    from mechbench_compute.lexicon import kinds as K

    return K.items_of(traj)


def _header(traj: Mapping[str, Any]) -> dict[str, Any]:
    """A trajectory collection's header: everything but the container
    fields and the items (or the retired `rows`)."""
    return {k: v for k, v in traj.items()
            if k not in ("kind", "item_kind", "key", "items", "rows")}


def project(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """trajectory/project — every row's scalar coordinate
    along a direction. The direction's layer is recorded, not enforced:
    projecting a layers-axis trajectory onto a single-layer direction is
    the funnel read against one axis, which is a legitimate question."""
    from mechbench_compute import directions as dirs

    traj = _trajectory_of(inputs.get("trajectory"))
    d = inputs.get("direction")
    if not isinstance(d, Mapping) or "vector" not in d:
        raise ValueError("trajectory/project needs a `direction` record")
    dv = dirs.as_array(d)
    if dv.shape[0] != int(traj.get("d_model", dv.shape[0])):
        raise ValueError(
            f"direction has {dv.shape[0]} dims; the trajectory has "
            f"{traj.get('d_model')}")
    keep = bool(params.get("keep_vectors", False))
    rows = []
    for r in _rows(traj):
        v = np.asarray(r["vector"], dtype=np.float32)
        item = S.coordinate(
            float(v @ dv), S.space_of(r, header=traj), d,
            id=r.get("id"), coords=S.coords_of(r), token=r.get("token"),
            step=r.get("step"), position=r.get("position"),
            n_pooled=r.get("n_pooled"), steps=r.get("steps"))
        if keep:
            item["vector"] = r["vector"]
        rows.append(item)
    from mechbench_compute.lexicon import kinds as K

    header = _header(traj)
    header["projected"] = True
    return K.collection("activations/coordinate", rows, **header)


def compare(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """trajectory/compare — trajectories `a` and `b`,
    paired by (id, step) (or by step alone with `pair_by: "step"`, for
    two single-item trajectories under different prompts or models):
    per-step cosine, angle in degrees, norm ratio; and the DIVERGENCE
    step — the first at which cosine falls below `threshold`."""
    a = _trajectory_of(inputs.get("a"), "a")
    b = _trajectory_of(inputs.get("b"), "b")
    if a.get("axis") != b.get("axis"):
        raise ValueError("trajectories must share an axis to be compared")
    pair_by = str(params.get("pair_by", "id"))
    threshold = float(params.get("threshold", 0.9))

    def key(r):
        return (r.get("id"), r["step"]) if pair_by == "id" else (r["step"],)

    bm = {key(r): r for r in _rows(b)}
    rows = []
    for r in _rows(a):
        s = bm.get(key(r))
        if s is None:
            continue
        va = np.asarray(r["vector"], dtype=np.float32)
        vb = np.asarray(s["vector"], dtype=np.float32)
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        cos = float(va @ vb / (na * nb)) if na and nb else 0.0
        cos = max(-1.0, min(1.0, cos))
        rows.append({
            "id": r.get("id") if pair_by == "id" else None,
            "step": r["step"], "layer": r.get("layer"),
            "position": r.get("position"),
            "cosine": round(cos, 6),
            "angle_deg": round(math.degrees(math.acos(cos)), 4),
            "norm_a": round(na, 5), "norm_b": round(nb, 5),
            "norm_ratio": round(nb / na, 6) if na else None,
        })
    if not rows:
        raise ValueError("no rows paired — do the trajectories share ids/steps?")
    # Per-step summary across ids, then the divergence step over it.
    by_step: dict[int, list[float]] = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r["cosine"])
    steps = sorted(by_step)
    mean_cos = {s: float(np.mean(by_step[s])) for s in steps}
    diverge = next((s for s in steps if mean_cos[s] < threshold), None)
    from mechbench_compute.lexicon import kinds as K

    return K.collection(
        "trajectory/comparison", rows,
        axis=a.get("axis"),
        pair_by=pair_by,
        threshold=threshold,
        n_pairs=len(rows),
        divergence_step=diverge,
        min_cosine_step=min(steps, key=lambda s: mean_cos[s]),
        per_step=[{"step": s, "mean_cosine": round(mean_cos[s], 6),
                   "n": len(by_step[s])} for s in steps],
    )


def aggregate(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """trajectory/aggregate — group rows and reduce.

    by:      "label" (default) | "id" | a coord/field name on the rows
    steps:   "all" | {"range": [a, b]} — which steps enter the reduce
    as:      "per_step" (default): mean per (group, step) with spread —
                 vectors → mean vector, mean norm, spread (mean cosine
                 to the mean); coords → mean, std.
             "window":  ONE value per group over the step window —
                 coords → mean/std/n (014's late-window commitment per
                 story, with by: "id"); vectors → mean vector.
             "vectors": one `residual_vectors` row per group (mean over
                 items and steps in the window), labelled by the group —
                 what `direction/from-vectors` reads, so an outcome axis
                 is this block followed by that one.
    """
    traj = _trajectory_of(inputs.get("trajectory"),
                          coords_ok=True)
    by = str(params.get("by", "label"))
    mode = str(params.get("as", "per_step"))
    if mode not in ("per_step", "window", "vectors"):
        raise ValueError("`as` must be 'per_step', 'window' or 'vectors'")
    steps = params.get("steps", "all")
    lo, hi = (None, None)
    if isinstance(steps, Mapping) and "range" in steps:
        lo, hi = int(steps["range"][0]), int(steps["range"][1])
    rows = _rows(traj)
    scalar = bool(rows) and "coord" in rows[0]
    if mode == "vectors" and scalar:
        raise ValueError("as: 'vectors' needs vector rows, not a projection")

    def group_of(r):
        # A coordinate first (`by: "genre"`), `id`, then a field on the
        # item; the retired `label` field is the `label` coordinate.
        if by == "id":
            return r.get("id")
        g = S.label_of(r, by)
        return g if g is not None else r.get(by)

    groups: dict[Any, dict[int, list]] = {}
    for r in rows:
        s = int(r["step"])
        if lo is not None and not (lo <= s < hi):
            continue
        g = group_of(r)
        val = r["coord"] if scalar else np.asarray(r["vector"], dtype=np.float32)
        groups.setdefault(g, {}).setdefault(s, []).append(val)
    if not groups:
        raise ValueError("no rows to aggregate (empty window or no groups)")

    out_rows: list[dict[str, Any]] = []
    if mode == "per_step":
        for g, by_s in groups.items():
            for s in sorted(by_s):
                vals = by_s[s]
                if scalar:
                    a = np.asarray(vals, dtype=np.float64)
                    out_rows.append({"group": g, "step": s, "n": len(vals),
                                     "mean": round(float(a.mean()), 6),
                                     "std": round(float(a.std()), 6)})
                else:
                    m = np.mean(np.stack(vals), axis=0)
                    mn = float(np.linalg.norm(m))
                    cos = [float(v @ m / (np.linalg.norm(v) * mn))
                           for v in vals if mn and np.linalg.norm(v)]
                    out_rows.append({
                        "group": g, "step": s, "n": len(vals),
                        "norm": round(mn, 5),
                        "mean_norm": round(float(np.mean(
                            [np.linalg.norm(v) for v in vals])), 5),
                        "spread": round(float(np.mean(cos)), 6) if cos else None,
                        "vector": [round(float(x), 5) for x in m],
                    })
        from mechbench_compute.lexicon import kinds as K

        return K.collection("trajectory/summary", out_rows, **_header(traj),
                            aggregated={"by": by, "as": mode, "steps": steps})

    # window / vectors: one value per group over everything in the window
    first_space = S.space_of(rows[0], header=traj) if rows else None
    layer = first_space.get("layer") if first_space else None
    for g, by_s in groups.items():
        vals = [v for s in by_s for v in by_s[s]]
        if scalar:
            a = np.asarray(vals, dtype=np.float64)
            out_rows.append({"group": g, "n": len(vals),
                             "mean": round(float(a.mean()), 6),
                             "std": round(float(a.std()), 6)})
        else:
            m = np.mean(np.stack(vals), axis=0)
            # The group goes out as a string coordinate on the `by` axis:
            # `direction/from-vectors` names its groups as strings, and a
            # text/stats hit arrives as the integer 1/0.
            out_rows.append(S.vector(m, first_space, id=str(g), coords={by: str(g)},
                                     n_pooled=len(vals)))
    from mechbench_compute.lexicon import kinds as K

    if mode == "vectors":
        return K.collection(
            "activations/vector", out_rows,
            model=first_space.get("model") if first_space else None,
            point=traj.get("point"),
            source="resid",
            position=f"trajectory-window {steps}",
            layers=[layer] if layer is not None else traj.get("layers"),
            d_model=traj.get("d_model"),
            template=traj.get("template"),
        )
    return K.collection("trajectory/summary", out_rows, **_header(traj),
                        aggregated={"by": by, "as": mode, "steps": steps})


#: Pure blocks this module contributes (registered by blocks.PURE_BLOCKS).
PURE: dict[str, Callable[..., Any]] = {
    "trajectory/project": project,
    "trajectory/compare": compare,
    "trajectory/aggregate": aggregate,
}
