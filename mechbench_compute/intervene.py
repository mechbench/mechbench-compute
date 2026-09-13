"""The declarative `intervene` block (task 000366, epic 000364).

Intervention is a product of two small sets. POINTS: any hook point of
the forward pass. OPERATIONS: zero, mean, resample, patch, add, scale,
clamp, project_out, rotate — over sets of positions, heads and
neurons, optionally conditioned on a direction's projection or on
token identity. One spec list, applied together in one forward, then
a readout (the next-token distribution, or captures). The old
ablate/steer blocks are special cases of this grammar.

A spec item::

    {"point": "resid_post", "layers": [14] | "all",
     "positions": "last" | "all" | [3, 5] | {"tokens": ["lighthouse"]} | {"range": [2, 6]},
     "heads": [0, 3] | null, "neurons": [17, 902] | null,
     "op": "add", "strength": 4.0,
     "direction": <direction object>, "direction2": <direction> (rotate),
     "source": <residual_vectors record> (mean / resample / patch), "row": {...},
     "condition": {"direction": <direction>, "threshold": 0.0, "above": true}}

Everything here is deterministic given the spec and the records
(`resample` draws from `seed`), so the block is `reproducible`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from mechbench_compute import directions as dirs

OPS = ("zero", "mean", "resample", "patch", "add", "scale", "clamp",
       "project_out", "rotate")

#: point family -> (position axis, head axis, feature axis) of the tensor
#: dispatched at that point. Tensors are batch-first.
_LAYOUT: dict[str, tuple[int, int | None, int]] = {
    # [B, L, D]
    "resid_pre": (1, None, 2), "resid_post": (1, None, 2), "attn_out": (1, None, 2),
    "mlp_out": (1, None, 2), "gate_out": (1, None, 2), "attn.in_norm": (1, None, 2),
    "mlp.in_norm": (1, None, 2), "embed": (1, None, 2), "final_norm": (1, None, 2),
    # [B, L, F]  (neurons on the feature axis)
    "mlp.gate": (1, None, 2), "mlp.up": (1, None, 2), "mlp.act": (1, None, 2),
    "mlp.down_in": (1, None, 2),
    # [B, L, V]
    "logits": (1, None, 2),
    # [B, L, n_heads*hd]
    "attn.o_in": (1, None, 2),
    # [B, n_heads, L, hd]
    "attn.q": (2, 1, 3), "attn.q_pre_rope": (2, 1, 3), "attn.per_head_out": (2, 1, 3),
    # [B, n_kv, L_kv, hd]
    "attn.k": (2, 1, 3), "attn.v": (2, 1, 3), "attn.k_pre_rope": (2, 1, 3),
    # [B, L, n_heads, hd]  (pre-transpose)
    "attn.q_pre_norm": (1, 2, 3), "attn.k_pre_norm": (1, 2, 3),
    # [B, n_heads, L, S]  (query positions on axis 2; keys are the feature axis)
    "attn.weights": (2, 1, 3), "attn.scores": (2, 1, 3),
}

_GLOBAL_POINTS = frozenset({"embed", "final_norm", "logits"})


class SpecError(ValueError):
    pass


# --- parsing -----------------------------------------------------------------------


def _as_list(x: Any) -> list[int]:
    if isinstance(x, int):
        return [x]
    return [int(i) for i in x]


def _rows_matrix(source: Mapping[str, Any], layer: int | None) -> np.ndarray:
    if not isinstance(source, Mapping) or source.get("kind") != "residual_vectors":
        raise SpecError("`source` must be a residual_vectors record")
    rows = [r for r in source.get("rows", [])
            if layer is None or r.get("layer") == layer]
    if not rows:
        raise SpecError(f"`source` has no rows at layer {layer}")
    return np.array([r["vector"] for r in rows], dtype=np.float32)


class Spec:
    """One parsed spec item; `build(ids, tokens, layer)` returns the hook fn."""

    def __init__(self, item: Mapping[str, Any], *, n_layers: int, seed: int) -> None:
        point = str(item.get("point", "resid_post"))
        if point not in _LAYOUT:
            raise SpecError(f"unknown or unsupported point {point!r}; "
                            f"one of {sorted(_LAYOUT)}")
        op = str(item.get("op", "zero"))
        if op not in OPS:
            raise SpecError(f"unknown op {op!r}; one of {OPS}")
        self.point, self.op = point, op
        layers = item.get("layers", "all")
        if point in _GLOBAL_POINTS:
            self.layers: list[int | None] = [None]
        elif layers == "all":
            self.layers = list(range(n_layers))
        else:
            self.layers = _as_list(layers)
            for layer in self.layers:
                if not 0 <= int(layer) < n_layers:
                    raise SpecError(f"layer {layer} out of range (n_layers={n_layers})")
        self.positions = item.get("positions", "last")
        self.heads = None if item.get("heads") is None else _as_list(item["heads"])
        self.neurons = None if item.get("neurons") is None else _as_list(item["neurons"])
        self.strength = float(item.get("strength", 1.0))
        self.seed = int(item.get("seed", seed))
        self.direction = (dirs.as_array(item["direction"])
                          if item.get("direction") is not None else None)
        self.direction2 = (dirs.as_array(item["direction2"])
                           if item.get("direction2") is not None else None)
        self.source = item.get("source")
        self.row = item.get("row")
        cond = item.get("condition")
        self.condition = None
        if cond:
            self.condition = (dirs.as_array(cond["direction"]),
                              float(cond.get("threshold", 0.0)),
                              bool(cond.get("above", True)))
        if op in ("add", "project_out", "clamp", "rotate") and self.direction is None:
            raise SpecError(f"op {op!r} needs a `direction`")
        if op == "rotate" and self.direction2 is None:
            raise SpecError("op 'rotate' needs `direction2` (the plane's second axis)")
        if op in ("mean", "resample", "patch") and self.source is None and op != "patch":
            raise SpecError(f"op {op!r} needs a `source` residual_vectors record")
        if op == "patch" and self.source is None and self.direction is None:
            raise SpecError("op 'patch' needs a `source` record (with `row`) or a `direction`")
        self._rng = np.random.default_rng(self.seed)

    # ---- hook construction --------------------------------------------------------

    def hook_names(self) -> list[str]:
        return [self.point if layer is None else f"blocks.{layer}.{self.point}"
                for layer in self.layers]

    def build(self, layer: int | None, tokens: Sequence[str]) -> Callable:
        pos_axis, head_axis, feat_axis = _LAYOUT[self.point]
        positions = self.positions
        heads, neurons = self.heads, self.neurons
        op, strength = self.op, self.strength
        d = None if self.direction is None else mx.array(self.direction)
        cond = self.condition
        # per-position replacement rows for mean / resample / patch
        rows = None
        if self.source is not None:
            rows = _rows_matrix(self.source, layer)
        rng = self._rng

        def _positions(L: int) -> list[int]:
            if positions == "all":
                return list(range(L))
            if positions == "last":
                return [L - 1]
            if isinstance(positions, Mapping):
                if "range" in positions:
                    a, b = positions["range"]
                    return list(range(max(0, int(a)), min(L, int(b))))
                if "tokens" in positions:
                    want = {str(t).strip().casefold() for t in positions["tokens"]}
                    hit = [i for i, t in enumerate(tokens[:L])
                           if str(t).strip().casefold() in want]
                    if not hit:
                        raise SpecError(f"none of {sorted(want)} among the prompt's tokens")
                    return hit
                raise SpecError(f"unknown positions spec {positions!r}")
            return [(int(p) + L) % L for p in _as_list(positions)]

        def fn(act: mx.array, info) -> mx.array:
            shape = act.shape
            nd = len(shape)
            L = shape[pos_axis]
            sel_pos = _positions(L)

            def axis_mask(axis: int, idx: Sequence[int]) -> mx.array:
                m = np.zeros(shape[axis], dtype=bool)
                m[list(idx)] = True
                view = [1] * nd
                view[axis] = shape[axis]
                return mx.array(m).reshape(view)

            mask = axis_mask(pos_axis, sel_pos)
            if heads is not None:
                if head_axis is None:
                    raise SpecError(f"point {self.point!r} has no head axis")
                mask = mask & axis_mask(head_axis, heads)
            if neurons is not None:
                mask = mask & axis_mask(feat_axis, neurons)

            def along_feat(v: mx.array) -> mx.array:
                view = [1] * nd
                view[feat_axis] = shape[feat_axis]
                if v.size != shape[feat_axis]:
                    raise SpecError(
                        f"direction width {v.size} does not match the feature axis "
                        f"({shape[feat_axis]}) at point {self.point!r}")
                return v.reshape(view).astype(act.dtype)

            if cond is not None:
                cd, thr, above = cond
                proj = mx.sum(act * along_feat(mx.array(cd)), axis=feat_axis, keepdims=True)
                cmask = (proj > thr) if above else (proj < thr)
                mask = mask & cmask

            if op == "zero":
                new = mx.zeros_like(act)
            elif op == "scale":
                new = act * strength
            elif op == "add":
                new = act + strength * along_feat(d)
            elif op == "project_out":
                dd = along_feat(d)
                p = mx.sum(act * dd, axis=feat_axis, keepdims=True)
                new = act - strength * p * dd
            elif op == "clamp":
                dd = along_feat(d)
                p = mx.sum(act * dd, axis=feat_axis, keepdims=True)
                clipped = mx.clip(p, -abs(strength), abs(strength))
                new = act + (clipped - p) * dd
            elif op == "rotate":
                u = along_feat(d)
                w0 = mx.array(self.direction2 - float(self.direction2 @ self.direction) * self.direction)
                w0 = w0 / mx.maximum(mx.sqrt(mx.sum(w0 * w0)), 1e-8)
                w = along_feat(w0)
                a_u = mx.sum(act * u, axis=feat_axis, keepdims=True)
                a_w = mx.sum(act * w, axis=feat_axis, keepdims=True)
                c, s = math.cos(strength), math.sin(strength)
                new = act - a_u * u - a_w * w + (a_u * c - a_w * s) * u + (a_u * s + a_w * c) * w
            elif op in ("mean", "resample", "patch"):
                if rows is not None:
                    if op == "mean":
                        vec = rows.mean(0)
                    elif op == "resample":
                        vec = rows[int(rng.integers(len(rows)))]
                    else:
                        r = self.row or {}
                        vec = rows[int(r.get("index", 0))]
                else:
                    vec = np.asarray(self.direction, dtype=np.float32)
                new = mx.broadcast_to(along_feat(mx.array(vec)), shape)
            else:  # pragma: no cover
                raise SpecError(op)
            return mx.where(mask, new, act)

        return fn


class SpecIntervention:
    """An `Intervention` (as_hooks / as_captures) over a whole spec list
    for one record's tokens."""

    def __init__(self, specs: Sequence[Spec], tokens: Sequence[str]) -> None:
        self._hooks: dict[str, Callable] = {}
        for spec in specs:
            for layer, name in zip(spec.layers, spec.hook_names(), strict=True):
                fn = spec.build(layer, tokens)
                prev = self._hooks.get(name)
                if prev is None:
                    self._hooks[name] = fn
                else:
                    def chained(act, info, _a=prev, _b=fn):
                        out = _a(act, info)
                        return _b(out if out is not None else act, info)
                    self._hooks[name] = chained

    def as_hooks(self) -> dict[str, Callable]:
        return dict(self._hooks)

    def as_captures(self) -> list[str]:
        return []


# --- the block ---------------------------------------------------------------------


def _wire_spec(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The spec as lineage records it: objects replaced by their provenance."""
    out = []
    for it in items:
        w = dict(it)
        for k in ("direction", "direction2"):
            if isinstance(w.get(k), Mapping):
                w[k] = {"kind": "direction", "layer": w[k].get("layer"),
                        "point": w[k].get("point"),
                        "derivation": w[k].get("derivation")}
        if isinstance(w.get("source"), Mapping):
            w["source"] = {"kind": w["source"].get("kind"),
                           "n_rows": len(w["source"].get("rows", []))}
        if isinstance(w.get("condition"), Mapping) and isinstance(w["condition"].get("direction"), Mapping):
            w["condition"] = {**w["condition"], "direction": {"derivation": w["condition"]["direction"].get("derivation")}}
        out.append(w)
    return out


def run(model, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
        on_item: Callable | None = None,
        on_start: Callable[[int], None] | None = None) -> dict[str, Any]:
    from mechbench_compute.interp import (
        _last_logp,
        _prompt_of,
        _target_token_id,
        _tokenize,
    )

    inputs = inputs or {}
    items = list(params.get("spec") or [])
    if not items:
        raise SpecError("intervene needs a non-empty `spec` list")
    # Objects may arrive by edge: a `direction` / `source` port fills any
    # spec item that names none of its own.
    port_dir = inputs.get("direction")
    port_src = inputs.get("source") or inputs.get("vectors")
    filled = []
    for it in items:
        it = dict(it)
        if it.get("direction") is None and port_dir is not None:
            it["direction"] = port_dir
        if it.get("source") is None and port_src is not None and it.get("op") in ("mean", "resample", "patch"):
            it["source"] = port_src
        filled.append(it)
    seed = int(params.get("seed", 0))
    specs = [Spec(it, n_layers=model.arch.n_layers, seed=seed) for it in filled]
    template = str(params.get("template", "raw"))
    factors = [float(f) for f in (params.get("sweep") or {}).get("strength", [1.0])]
    if bool(params.get("control", True)) and 0.0 not in factors:
        factors = [0.0, *factors]
    readout = dict(params.get("readout") or {"kind": "decision"})
    rk = str(readout.get("kind", "decision"))
    if rk not in ("decision", "capture"):
        raise SpecError(f"unknown readout kind {rk!r}")
    top_k = int(readout.get("top_k", params.get("top_k", 5)))
    if not records:
        raise SpecError("intervene needs at least one record")
    if on_start:
        on_start(len(records) * len(factors))

    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = _prompt_of(record)
        ids = _tokenize(model, prompt, template)
        flat = [int(t) for t in np.array(ids).reshape(-1)]
        tokens = [model.tokenizer.decode([t]) for t in flat]
        track = record.get("track") or params.get("track")
        track_id = _target_token_id(model, str(track)) if track else None
        outcomes = record.get("outcomes", params.get("outcomes"))
        for factor in factors:
            key = f"{record.get('id')}:{factor}"
            if factor == 0.0:
                ivs: list[Any] = []
            else:
                scaled = []
                for spec in specs:
                    s2 = spec
                    if factor != 1.0:
                        s2 = Spec.__new__(Spec)
                        s2.__dict__.update(spec.__dict__)
                        s2.strength = spec.strength * factor
                    scaled.append(s2)
                ivs = [SpecIntervention(scaled, tokens)]
            if rk == "decision":
                res = model.run(ids, interventions=ivs)
                lp = _last_logp(res.logits)
                probs = np.exp(lp.astype(np.float64))
                nz = probs[probs > 0]
                order = np.argsort(-lp)[:top_k]
                row: dict[str, Any] = {
                    "id": record.get("id"), "coords": dict(record.get("coords", {})),
                    "factor": factor,
                    "entropy_bits": round(float(-(nz * np.log2(nz)).sum()), 4),
                    "top": [{"token": model.tokenizer.decode([int(t)]),
                             "logp": round(float(lp[int(t)]), 3)} for t in order],
                }
                if track_id is not None:
                    row["track_logp"] = round(float(lp[track_id]), 3)
                if outcomes:
                    row["outcome_mass"] = {
                        str(o): round(float(probs[_target_token_id(model, str(o))]), 5)
                        for o in outcomes}
            else:
                points = [str(p) for p in readout.get("points", [])]
                if not points:
                    raise SpecError("capture readout needs `points`")
                res = model.run(ids, interventions=ivs, capture=points)
                L = len(flat)
                pos = readout.get("position", "final")
                pidx = L - 1 if pos in (None, "final") else (int(pos) + L) % L
                row = {"id": record.get("id"), "coords": dict(record.get("coords", {})),
                       "factor": factor, "position": pidx, "captures": {}}
                for p in points:
                    t = res.cache[p]
                    v = t[0, pidx] if t.ndim == 3 else t[0]
                    # bf16 has no numpy buffer protocol: cast first.
                    arr = np.array(v.astype(mx.float32)).reshape(-1)
                    row["captures"][p] = [round(float(x), 5) for x in arr[:4096]]
            rows.append(row)
            if on_item:
                on_item(key, row)
    return {
        "kind": "intervene_readout",
        "spec": _wire_spec(filled),
        "sweep": factors,
        "readout": rk,
        "template": template,
        "rows": rows,
        "description": (
            f"{len(specs)} intervention(s) applied together per forward; factor 0 "
            f"is the control; strengths scale with the sweep factor."),
    }
