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

A trajectory is rows of `{id, label, step, layer, position, vector,
norm}`, `step` indexing the axis. Four blocks:

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

# --- shapes ------------------------------------------------------------------

AXES = ("layers", "positions")


def _label_of(record: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    """`label` on the record, else the coord named by `label_coord`, else
    the top-level field named by `label_field` (what text/stats
    `annotate` writes — a pattern hit is a field, not a coord)."""
    label = record.get("label")
    if label is None and params.get("label_coord"):
        label = (record.get("coords") or record.get("metadata") or {}).get(
            params["label_coord"])
    if label is None and params.get("label_field"):
        label = record.get(params["label_field"])
    return label


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


def _position_indices(spec: Any, seq_len: int, gen_start: int | None,
                      prompt_len: int | None) -> list[int]:
    """Which positions a positions-axis trajectory reads.

    "all"                 every position
    "generated"           from where generation began (the trace's
                          span, else the tokenized prompt's end) —
                          the story, not the envelope
    {"range": [a, b]}     positions a..b-1 (b clipped to the sequence)
    {"after": n}          positions n.. (n clipped)
    """
    if spec in (None, "all"):
        return list(range(seq_len))
    if spec == "generated":
        start = gen_start if gen_start is not None else prompt_len
        if start is None:
            raise ValueError(
                "positions 'generated' needs a trace with generation_spans, "
                "or a record whose prompt length is known")
        return list(range(max(0, min(start, seq_len)), seq_len))
    if isinstance(spec, Mapping):
        if "range" in spec:
            a, b = spec["range"]
            return list(range(max(0, int(a)), min(int(b), seq_len)))
        if "after" in spec:
            return list(range(max(0, min(int(spec["after"]), seq_len)), seq_len))
    raise ValueError(f"unknown positions {spec!r}")


# --- capture (the model block) -----------------------------------------------


def capture(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """~canonical/ops/trajectory/capture/1."""
    import mlx.core as mx

    from mechbench_compute import Capture
    from mechbench_compute.interp import (
        MAX_VECTOR_FLOATS,
        _position_index,
        _prompt_of,
        _resolve_layers,
        _tokenize,
    )

    axis = str(params.get("axis", "layers"))
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}: one of {AXES}")
    point = str(params.get("point", "post"))
    template = str(params.get("template", "raw"))
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
    reduce = params.get("reduce")
    if reduce not in (None, "mean"):
        raise ValueError("reduce must be 'mean' when given")
    step_window = params.get("steps")
    lo, hi = (None, None)
    if isinstance(step_window, Mapping) and "range" in step_window:
        lo, hi = int(step_window["range"][0]), int(step_window["range"][1])
    direction = params.get("project")
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
        position = params.get("position", "final")
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
    elif reduce == "mean":
        per_record = width
    else:
        per_record = None if steps_per is None else steps_per * width
    if per_record:
        total = len(records) * per_record
        if total > MAX_VECTOR_FLOATS:
            raise ValueError(
                f"{len(records)} records × {per_record // width} steps × "
                f"{width} dims = {total} floats exceeds the "
                f"{MAX_VECTOR_FLOATS} cap — set `reduce`, `project`, or "
                "`max_steps`, or capture fewer records")
    if on_start:
        on_start(len(records))

    cap = Capture.residual(layers, point=point)
    tok = model.tokenizer
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
            ids = _tokenize(model, _prompt_of(record), template)
            prompt_len = None
        else:
            used_trace += 1
            ids = mx.array([ids_list])
            prompt_len = gen_start
        arr = np.array(ids).reshape(-1)
        seq_len = int(arr.shape[0])
        result = model.run(ids, interventions=[cap])
        label = _label_of(record, params)

        if axis == "layers":
            pos = _position_index(model, ids, record, position)
            for step, layer in enumerate(layers):
                t = result.cache[f"blocks.{layer}.resid_{point}"]
                v = t[0, pos, :].astype(mx.float32)
                mx.eval(v)
                rows.append(_row(record, label, step, layer, pos, np.array(v),
                                 tok, arr, model, vocab_top))
        else:
            layer = layers[0]
            t = result.cache[f"blocks.{layer}.resid_{point}"]
            seq = t[0].astype(mx.float32)
            mx.eval(seq)
            mat = np.array(seq)
            idx = _position_indices(position, seq_len, gen_start, prompt_len)
            if lo is not None:
                # `steps` counts from the trajectory's own start.
                idx = [p for s, p in enumerate(idx) if lo <= s < hi]
            if max_steps:
                idx = idx[: int(max_steps)]
            if not idx:
                raise ValueError(
                    f"record {record.get('id')!r}: no positions in the window")
            if reduce == "mean":
                v = mat[idx].mean(axis=0)
                row = _row(record, label, 0, layer, idx[0], v, tok, arr, model, 0)
                row.update({"token": None, "n_pooled": len(idx),
                            "steps": [int(lo or 0), int(hi) if hi else len(idx)]})
                rows.append(_projected(row, v, dvec))
            else:
                if dvec is None:
                    running = len(rows) + len(idx)
                    if running * width > MAX_VECTOR_FLOATS:
                        raise ValueError(
                            f"trajectory exceeds the {MAX_VECTOR_FLOATS}-float "
                            f"cap at record {record.get('id')!r}; set `max_steps`, "
                            "`reduce`, or `project`, or capture fewer records")
                for step, p in enumerate(idx):
                    row = _row(record, label, step, layer, p, mat[p], tok, arr,
                               model, vocab_top)
                    rows.append(_projected(row, mat[p], dvec))
        if on_item:
            on_item()

    prov = (direction.get("derivation") or {}) if isinstance(direction, Mapping) else {}
    return {
        "kind": "trajectory_projection" if dvec is not None else "trajectory",
        "axis": axis,
        "point": point,
        "layers": layers,
        "position": str(position) if axis == "layers" else None,
        "positions": (position if axis == "positions" else None),
        "d_model": width,
        "template": template,
        "replay": "trace" if used_trace == len(records)
                  else ("text" if used_trace == 0 else "mixed"),
        "n_items": len(records),
        **({"max_steps": int(max_steps)} if max_steps else {}),
        **({"reduce": reduce, "steps": step_window} if reduce else {}),
        **({"direction": {"layer": direction.get("layer"),
                          "point": direction.get("point"),
                          "method": prov.get("method"),
                          **({"labels": prov["labels"]} if prov.get("labels") else {})}}
           if dvec is not None else {}),
        "rows": rows,
    }


def _row(record, label, step, layer, pos, vec: np.ndarray, tok, arr, model,
         vocab_top: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": record.get("id"),
        "label": label,
        "step": int(step),
        "layer": int(layer),
        "position": int(pos),
        "token": tok.decode([int(arr[pos])]) if pos < len(arr) else None,
        "norm": round(float(np.linalg.norm(vec)), 5),
        "vector": [round(float(x), 5) for x in vec],
    }
    if vocab_top:
        row["vocab_top"] = _vocab_top(model, vec, vocab_top)
    return row


def _projected(row: dict[str, Any], vec: np.ndarray,
               dvec: np.ndarray | None) -> dict[str, Any]:
    """With a direction, a row is its scalar coordinate and carries no
    vector — the trace itself, small enough to be an object."""
    if dvec is None:
        return row
    out = {k: v for k, v in row.items() if k != "vector"}
    out["coord"] = round(float(np.asarray(vec, dtype=np.float32) @ dvec), 6)
    return out


def _vocab_top(model, vec: np.ndarray, k: int) -> list[dict[str, Any]]:
    """The vector through the unembedding: its top-k tokens with
    probabilities — the lens reading of this point."""
    import mlx.core as mx

    logits = model.project_to_logits(mx.array(vec)[None, :]).astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp)
    lp = np.array(lp).reshape(-1)
    top = np.argsort(-lp)[:k]
    return [{"token": model.tokenizer.decode([int(t)]),
             "p": round(float(math.exp(lp[t])), 6)} for t in top]


# --- pure blocks over trajectories -------------------------------------------


def _trajectory_of(x: Any, port: str = "trajectory") -> Mapping[str, Any]:
    if isinstance(x, Mapping) and x.get("kind") in ("trajectory",
                                                    "trajectory_projection"):
        return x
    raise ValueError(f"port {port!r} is not a trajectory record")


def project(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """~canonical/ops/trajectory/project/1 — every row's scalar coordinate
    along a direction. The direction's layer is recorded, not enforced:
    projecting a layers-axis trajectory onto a single-layer direction is
    the funnel read against one axis, which is a legitimate question."""
    from mechbench_compute import directions as dirs

    traj = _trajectory_of(inputs.get("trajectory") or params.get("trajectory"))
    d = inputs.get("direction") or params.get("direction")
    if not isinstance(d, Mapping) or "vector" not in d:
        raise ValueError("trajectory/project needs a `direction` record")
    dv = dirs.as_array(d)
    if dv.shape[0] != int(traj.get("d_model", dv.shape[0])):
        raise ValueError(
            f"direction has {dv.shape[0]} dims; the trajectory has "
            f"{traj.get('d_model')}")
    keep = bool(params.get("keep_vectors", False))
    rows = []
    for r in traj.get("rows", []):
        v = np.asarray(r["vector"], dtype=np.float32)
        row = {k: v_ for k, v_ in r.items() if k != "vector" or keep}
        row["coord"] = round(float(v @ dv), 6)
        rows.append(row)
    out = {k: v for k, v in traj.items() if k != "rows"}
    prov = d.get("derivation") or {}
    out.update({
        "kind": "trajectory_projection",
        # What it was projected onto, as the direction record says it:
        # method and labels ride in the direction's derivation.
        "direction": {"layer": d.get("layer"), "point": d.get("point"),
                      "method": prov.get("method"),
                      **({"labels": prov["labels"]} if prov.get("labels") else {})},
        "rows": rows,
    })
    return out


def compare(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """~canonical/ops/trajectory/compare/1 — trajectories `a` and `b`,
    paired by (id, step) (or by step alone with `pair_by: "step"`, for
    two single-item trajectories under different prompts or models):
    per-step cosine, angle in degrees, norm ratio; and the DIVERGENCE
    step — the first at which cosine falls below `threshold`."""
    a = _trajectory_of(inputs.get("a") or params.get("a"), "a")
    b = _trajectory_of(inputs.get("b") or params.get("b"), "b")
    if a.get("axis") != b.get("axis"):
        raise ValueError("trajectories must share an axis to be compared")
    pair_by = str(params.get("pair_by", "id"))
    threshold = float(params.get("threshold", 0.9))

    def key(r):
        return (r.get("id"), r["step"]) if pair_by == "id" else (r["step"],)

    bm = {key(r): r for r in b.get("rows", [])}
    rows = []
    for r in a.get("rows", []):
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
    return {
        "kind": "trajectory_comparison",
        "axis": a.get("axis"),
        "pair_by": pair_by,
        "threshold": threshold,
        "n_pairs": len(rows),
        "divergence_step": diverge,
        "min_cosine_step": min(steps, key=lambda s: mean_cos[s]),
        "per_step": [{"step": s, "mean_cosine": round(mean_cos[s], 6),
                      "n": len(by_step[s])} for s in steps],
        "rows": rows,
    }


def aggregate(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """~canonical/ops/trajectory/aggregate/1 — group rows and reduce.

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
    traj = _trajectory_of(inputs.get("trajectory") or params.get("trajectory"))
    by = str(params.get("by", "label"))
    mode = str(params.get("as", "per_step"))
    if mode not in ("per_step", "window", "vectors"):
        raise ValueError("`as` must be 'per_step', 'window' or 'vectors'")
    steps = params.get("steps", "all")
    lo, hi = (None, None)
    if isinstance(steps, Mapping) and "range" in steps:
        lo, hi = int(steps["range"][0]), int(steps["range"][1])
    rows = traj.get("rows", [])
    scalar = bool(rows) and "coord" in rows[0]
    if mode == "vectors" and scalar:
        raise ValueError("as: 'vectors' needs vector rows, not a projection")

    def group_of(r):
        return r.get(by) if by in r else r.get("label")

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
        kind = "trajectory_summary" if scalar else "trajectory"
        return {**{k: v for k, v in traj.items() if k != "rows"},
                "kind": kind, "aggregated": {"by": by, "as": mode,
                                             "steps": steps},
                "rows": out_rows}

    # window / vectors: one value per group over everything in the window
    layer = rows[0].get("layer") if rows else None
    for g, by_s in groups.items():
        vals = [v for s in by_s for v in by_s[s]]
        if scalar:
            a = np.asarray(vals, dtype=np.float64)
            out_rows.append({"group": g, "n": len(vals),
                             "mean": round(float(a.mean()), 6),
                             "std": round(float(a.std()), 6)})
        else:
            m = np.mean(np.stack(vals), axis=0)
            # Labels go out as strings: `direction/from-vectors` names its
            # positive/negative labels as strings, and a text/stats hit
            # arrives as the integer 1/0.
            out_rows.append({"id": str(g), "label": str(g), "layer": layer,
                             "n_pooled": len(vals),
                             "vector": [round(float(x), 5) for x in m]})
    if mode == "vectors":
        return {
            "kind": "residual_vectors",
            "point": traj.get("point"),
            "source": "resid",
            "position": f"trajectory-window {steps}",
            "layers": [layer] if layer is not None else traj.get("layers"),
            "d_model": traj.get("d_model"),
            "template": traj.get("template"),
            "rows": out_rows,
        }
    return {**{k: v for k, v in traj.items() if k != "rows"},
            "kind": "trajectory_summary" if scalar else "trajectory",
            "aggregated": {"by": by, "as": mode, "steps": steps},
            "rows": out_rows}


#: Pure blocks this module contributes (registered by blocks.PURE_BLOCKS).
PURE: dict[str, Callable[..., Any]] = {
    "~canonical/ops/trajectory/project/1": project,
    "~canonical/ops/trajectory/compare/1": compare,
    "~canonical/ops/trajectory/aggregate/1": aggregate,
}
