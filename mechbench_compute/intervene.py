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
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute.points import LAYOUT as _LAYOUT

OPS = ("zero", "mean", "resample", "patch", "add", "scale", "clamp",
       "project_out", "rotate")

_GLOBAL_POINTS = frozenset({"embed", "final_norm", "logits"})


class SpecError(ValueError):
    pass


# --- parsing -----------------------------------------------------------------------


def _as_list(x: Any) -> list[int]:
    if isinstance(x, int):
        return [x]
    return [int(i) for i in x]


def _hook_space(name: str) -> tuple[int | None, str]:
    """`blocks.14.resid_post` → (14, "resid_post"); a whole-model point
    → (None, name)."""
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) >= 3 and parts[1].isdigit():
        return int(parts[1]), ".".join(parts[2:])
    return None, name


def _source_items(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The vector items a `source` offers: a collection of
    activations/vector as it is, or every vector captured by a capture
    readout (`intervene/readout` items' `captures`), so one intervention's
    capture is another's source."""
    from mechbench_compute.lexicon import kinds as K

    if not isinstance(source, Mapping):
        raise SpecError("`source` must be a collection of activations/vector")
    ik = K.item_kind_of(source)
    if ik == "activations/vector":
        return list(K.items_of(source))
    if ik == "intervene/readout":
        # A capture readout stored before 0.110.0 nested its vectors under
        # each row's `captures`; since then a capture readout IS an
        # `activations/vector` collection and takes the branch above.
        out: list[Mapping[str, Any]] = []
        for item in K.items_of(source):
            caps = item.get("captures")
            if isinstance(caps, Mapping) and K.item_kind_of(caps) == "activations/vector":
                out.extend(K.items_of(caps))
        if out:
            return out
        raise SpecError("`source` is a readout collection with no captures")
    raise SpecError("`source` must be a collection of activations/vector "
                    "or a capture readout")


def _rows_matrix(source: Mapping[str, Any], layer: int | None,
                 point: str | None = None) -> np.ndarray:
    rows = [r for r in _source_items(source)
            if (layer is None or S.layer_of(r) == layer)
            and (point is None or S.space_of(r).get("point") == point
                 or "space" not in r)]
    if not rows:
        raise SpecError(f"`source` has no vectors at layer {layer}"
                        + (f", point {point!r}" if point else ""))
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

    def build(self, layer: int | None, tokens: Sequence[str],
              record: Mapping[str, Any] | None = None) -> Callable:
        pos_axis, head_axis, feat_axis = _LAYOUT[self.point]
        positions = self.positions
        heads, neurons = self.heads, self.neurons
        op, strength = self.op, self.strength
        d = None if self.direction is None else mx.array(self.direction)
        cond = self.condition
        # per-position replacement rows for mean / resample / patch
        rows = None
        if self.source is not None:
            try:
                rows = _rows_matrix(self.source, layer, self.point)
            except SpecError:
                # A source captured at another point still serves an op
                # that only needs vectors of the right width.
                rows = _rows_matrix(self.source, layer)
        rng = self._rng

        def _positions(L: int) -> list[int]:
            # One grammar (positions.py); `L` is the tensor's own length at
            # this point, which a key/value axis may differ in.
            try:
                return POS.resolve(positions, L, tokens=list(tokens[:L]), record=record,
                                   prompt_len=min(len(tokens), L))
            except ValueError as e:
                raise SpecError(str(e)) from None

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

    def __init__(self, specs: Sequence[Spec], tokens: Sequence[str],
                 record: Mapping[str, Any] | None = None) -> None:
        self._hooks: dict[str, Callable] = {}
        for spec in specs:
            for layer, name in zip(spec.layers, spec.hook_names(), strict=True):
                fn = spec.build(layer, tokens, record)
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
                w[k] = {"kind": "direction/vector", "space": S.space_of(w[k]),
                        "derivation": w[k].get("derivation")}
        if isinstance(w.get("source"), Mapping):
            from mechbench_compute.lexicon import kinds as K

            src = w["source"]
            w["source"] = {"kind": src.get("kind"), "item_kind": K.item_kind_of(src),
                           "n_items": len(K.items_of(src)) if K.item_kind_of(src) else 0}
        if isinstance(w.get("condition"), Mapping) and isinstance(w["condition"].get("direction"), Mapping):
            w["condition"] = {**w["condition"], "direction": {"derivation": w["condition"]["direction"].get("derivation")}}
        out.append(w)
    return out


def _order(records: Sequence[Mapping[str, Any]], factors: Sequence[float],
           weight_items: Sequence[Mapping[str, Any]], model):
    """(record, factor) pairs, with any weight edits in scope.

    Without weight items this is the loop it always was: record outer,
    factor inner. With them the factor goes outside, because a weight
    edit is applied once for every record that runs under it — and the
    edit is undone before the next factor, and before the generator
    returns, whatever happens in between. A run that left a model edited
    would poison every later node in the job, which is the failure this
    `finally` exists for.
    """
    if not weight_items:
        for record in records:
            for factor in factors:
                yield record, factor
        return

    from mechbench_compute import weights as weights_mod

    for factor in factors:
        handle = ([] if factor == 0.0
                  else weights_mod.edit_parameters(model.lm, weight_items, factor))
        try:
            for record in records:
                yield record, factor
        finally:
            weights_mod.restore_parameters(model.lm, handle)


def run(model, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
        on_item: Callable | None = None,
        on_start: Callable[[int], None] | None = None) -> dict[str, Any]:
    from mechbench_compute.distill import render
    from mechbench_compute.interp import _last_logp

    inputs = inputs or {}
    items = list(params.get("spec") or [])
    if not items:
        raise SpecError("intervene needs a non-empty `spec` list")
    # Objects may arrive by edge: a `direction` / `source` port fills any
    # spec item that names none of its own.
    port_dir = inputs.get("direction")
    port_src = inputs.get("source")
    filled = []
    for it in items:
        it = dict(it)
        if it.get("direction") is None and port_dir is not None:
            it["direction"] = port_dir
        if it.get("source") is None and port_src is not None and it.get("op") in ("mean", "resample", "patch"):
            it["source"] = port_src
        filled.append(it)
    seed = int(params.get("seed", 0))
    # An item that names a `parameter` edits a WEIGHT, not an activation
    # (task 000457). Its scope is the node rather than the forward pass:
    # the tensor is changed, every record runs against the changed model,
    # and the original is reinstalled afterwards. The two kinds compose —
    # a spec may zero a weight and add a direction mid-forward — so they
    # are separated here and applied in their own scopes.
    weight_items = [it for it in filled if it.get("parameter") is not None]
    filled = [it for it in filled if it.get("parameter") is None]
    specs = [Spec(it, n_layers=model.arch.n_layers, seed=seed) for it in filled]
    if not specs and not weight_items:
        raise SpecError("intervene needs a non-empty `spec` list")
    factors = [float(f) for f in (params.get("sweep") or {}).get("strength", [1.0])]
    if bool(params.get("control", True)) and 0.0 not in factors:
        factors = [0.0, *factors]
    readout = dict(params.get("readout") or {"type": "decision"})
    rk = str(readout.get("type") or readout.get("kind") or "decision")
    if rk not in ("decision", "capture"):
        raise SpecError(f"unknown readout kind {rk!r}")
    top_k = int(readout.get("top_k", params.get("top_k", 5)))
    if not records:
        raise SpecError("intervene needs at least one record")
    if on_start:
        on_start(len(records) * len(factors))

    from mechbench_compute.interp import _tracked_ids
    from mechbench_compute.lexicon import kinds as K

    mid = S.model_id_of(model)
    rows: list[dict[str, Any]] = []
    # A weight edit is applied once per sweep factor, not once per
    # record: the tensor is the same for every record that runs under it,
    # and an SVD per record would be absurd. So the factor is the outer
    # loop when there are weight items, and the record loop is the same
    # body either way.
    for record, factor in _order(records, factors, weight_items, model):
            ids = render(model, record).array
            flat = [int(t) for t in np.array(ids).reshape(-1)]
            tokens = [model.tokenizer.decode([t]) for t in flat]
            tracked = _tracked_ids(model, record, tracked=params.get("tracked"))
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
                ivs = [SpecIntervention(scaled, tokens, record)]
            if rk == "decision":
                res = model.run(ids, interventions=ivs)
                lp = _last_logp(res.logits)
                row: dict[str, Any] = {
                    "id": record.get("id"), "coords": dict(record.get("coords", {})),
                    "factor": factor,
                    **S.distribution(lp, model.tokenizer, top_k=top_k, tracked=tracked),
                }
            else:
                points = [str(p) for p in readout.get("points", [])]
                if not points:
                    raise SpecError("capture readout needs `points`")
                res = model.run(ids, interventions=ivs, capture=points)
                L = len(flat)
                pidx = POS.one(readout.get("position", "last"), L, tokens=tokens,
                               record=record, prompt_len=L)
                # A capture under intervention IS a capture: one
                # `activations/vector` per hook point, the shape
                # `activations/capture` emits, so whatever reads a
                # capture — `geometry/compare`, `direction/regress`,
                # another intervention's `source` — reads this one too.
                # `factor` rides on each vector, since a sweep's rows
                # differ only by it (000599).
                for p in points:
                    t = res.cache[p]
                    v = t[0, pidx] if t.ndim == 3 else t[0]
                    # bf16 has no numpy buffer protocol: cast first.
                    arr = np.array(v.astype(mx.float32)).reshape(-1)[:4096]
                    cl, cp = _hook_space(p)
                    row = S.vector(
                        arr, S.space(model=mid, layer=cl, point=cp, d=int(arr.size)),
                        id=record.get("id"), coords=dict(record.get("coords", {})),
                        factor=factor, position=pidx,
                        token=S.token(model.tokenizer, flat[pidx]))
                    rows.append(row)
                    if on_item:
                        on_item(f"{key}:{p}", row)
                continue
            rows.append(row)
            if on_item:
                on_item(key, row)
    # The weight edits ride in the header beside the activation spec, so
    # a reader of the result knows the model was not the one on the shelf
    # (task 000457: "the manifest says so").
    weights_wire = [dict(it) for it in weight_items] or None
    what = []
    if specs:
        what.append(f"{len(specs)} activation intervention(s) per forward")
    if weight_items:
        what.append(f"{len(weight_items)} weight edit(s) for the run, "
                    f"restored after")
    return K.collection(
        "activations/vector" if rk == "capture" else "intervene/readout", rows,
        spec=_wire_spec(filled),
        weights=weights_wire,
        sweep=factors,
        readout=rk,
        **({"model": mid, "position": str(readout.get("position", "last")),
            "points": [str(p) for p in readout.get("points", [])]} if rk == "capture" else {}),
        description=(
            f"{'; '.join(what)}. Factor 0 is the control; strengths scale "
            f"with the sweep factor."),
    )
