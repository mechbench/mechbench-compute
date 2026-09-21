from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import directions as dirs
from mechbench_compute import positions as POS
from mechbench_compute._mlx import mx
from mechbench_compute.intervene.as_list import _as_list
from mechbench_compute.intervene.rows_matrix import _rows_matrix
from mechbench_compute.intervene.spec_error import SpecError
from mechbench_compute.points import LAYOUT as _LAYOUT

OPS = ("zero", "mean", "resample", "patch", "add", "scale", "clamp",
       "project_out", "rotate")


_GLOBAL_POINTS = frozenset({"embed", "final_norm", "logits"})


#: "the selector this item already names" — a sentinel, because None is
#: a selector (`positions: null` is `"last"`).
_SAME = object()


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
        # `except` inverts the sets this item names — everything BUT
        # these layers, heads or neurons — which is how a circuit's
        # completeness is measured against its faithfulness (000604).
        # Layers invert here, where the count is known; heads and
        # neurons invert in the hook, where the tensor's own axis says
        # how many there are.
        self.excepted = bool(item.get("except", False))
        if self.excepted:
            named = [k for k in ("layers", "heads", "neurons") if item.get(k) is not None]
            if point in _GLOBAL_POINTS:
                raise SpecError(
                    f"`except` inverts a set of layers, heads or neurons; point "
                    f"{point!r} has none — it occurs once per forward pass")
            if not named:
                raise SpecError(
                    "`except` needs a set to invert: name `layers`, `heads` or "
                    "`neurons` on the item")
            if item.get("layers") is not None and layers != "all":
                keep = {int(x) for x in self.layers}
                self.layers = [i for i in range(n_layers) if i not in keep]
        # A `patch` may take its row from ANOTHER layer or point — the
        # patchscope's move, reading a hidden state by writing it where
        # a different prompt would read it (000605).
        self.patch_from = item.get("from")
        if self.patch_from is not None:
            if op != "patch":
                raise SpecError("`from` names where a `patch` reads; this item's op is "
                                f"{op!r}")
            if not isinstance(self.patch_from, Mapping):
                raise SpecError("`from` is an object: {layer, point}")
        # An attention edge: which SOURCE positions the selected
        # destinations may attend to (000612).
        self.pattern = item.get("pattern")
        if self.pattern is not None:
            if point not in ("attn.scores", "attn.weights"):
                raise SpecError(
                    f"`pattern` names an attention edge; point {point!r} has no "
                    "source axis — one of 'attn.scores', 'attn.weights'")
            if not isinstance(self.pattern, Mapping) or "from" not in self.pattern:
                raise SpecError('`pattern` is `{"from": selector, "to": selector}`; '
                                "`from` is required")
        self.renormalize = bool(item.get("renormalize", True))
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
              record: Mapping[str, Any] | None = None,
              prompt_len: int | None = None, growing: bool = False) -> Callable:
        """The hook for one layer. `tokens` is the WHOLE sequence the
        positions resolve against — a list the caller may grow while
        decoding (000601) — and `prompt_len` where the prompt ends, for
        `"generated"`; None means the tokens as first given. `growing`
        says the sequence is still being written, so a token the
        selector names that has not arrived selects nothing rather than
        refusing; a one-shot pass keeps the refusal, which is how a
        misspelt token is caught."""
        pos_axis, head_axis, feat_axis = _LAYOUT[self.point]
        fixed_prompt_len = len(tokens) if prompt_len is None else int(prompt_len)
        absent = "none" if growing else "error"
        positions = self.positions
        heads, neurons = self.heads, self.neurons
        op, strength = self.op, self.strength
        d = None if self.direction is None else mx.array(self.direction)
        cond = self.condition
        # per-position replacement rows for mean / resample / patch
        rows = None
        if self.source is not None:
            src_layer, src_point = layer, self.point
            if self.patch_from is not None:
                src_layer = self.patch_from.get("layer", layer)
                src_point = self.patch_from.get("point", self.point)
            try:
                rows = _rows_matrix(self.source, src_layer, src_point)
            except SpecError:
                # A source captured at another point still serves an op
                # that only needs vectors of the right width.
                rows = _rows_matrix(self.source, src_layer)
        rng = self._rng

        def _positions(L: int, offset: int, selector: Any = _SAME) -> list[int]:
            # One grammar (positions.py), resolved over the whole sequence
            # seen so far — `offset` tokens already in the KV cache, then
            # this chunk's `L` — and kept only where it falls inside the
            # chunk, relative to it. A whole-prompt pass is the chunk at
            # offset 0. `L` is the tensor's own length at this point,
            # which a key/value axis may differ in.
            n = offset + L
            want = positions if selector is _SAME else selector
            try:
                sel = POS.resolve(want, n, tokens=list(tokens[:n]), record=record,
                                  prompt_len=min(fixed_prompt_len, n), absent=absent)
            except ValueError as e:
                raise SpecError(str(e)) from None
            return [p - offset for p in sel if offset <= p < n]

        def fn(act: mx.array, info) -> mx.array:
            shape = act.shape
            nd = len(shape)
            L = shape[pos_axis]
            sel_pos = _positions(L, int(getattr(info, "offset", 0) or 0))
            if not sel_pos:
                return act

            def axis_mask(axis: int, idx: Sequence[int], invert: bool = False) -> mx.array:
                m = np.zeros(shape[axis], dtype=bool)
                m[list(idx)] = True
                if invert:
                    m = ~m
                view = [1] * nd
                view[axis] = shape[axis]
                return mx.array(m).reshape(view)

            if self.pattern is not None:
                # An edge: the destinations attend along the position
                # axis, the sources sit on the feature (key) axis, which
                # counts the whole sequence and so takes no offset.
                to_sel = (sel_pos if "to" not in self.pattern
                          else _positions(L, int(getattr(info, "offset", 0) or 0), self.pattern["to"]))
                from_sel = _positions(shape[feat_axis], 0, self.pattern["from"])
                if not to_sel or not from_sel:
                    return act
                mask = axis_mask(pos_axis, to_sel) & axis_mask(feat_axis, from_sel)
            else:
                mask = axis_mask(pos_axis, sel_pos)
                if neurons is not None:
                    mask = mask & axis_mask(feat_axis, neurons, invert=self.excepted)
            if heads is not None:
                if head_axis is None:
                    raise SpecError(f"point {self.point!r} has no head axis")
                mask = mask & axis_mask(head_axis, heads, invert=self.excepted)

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
                # At a score, zero is a SCORE of zero, not a cut: what
                # removes an edge before the softmax is −inf (000612).
                new = (mx.full(shape, float("-inf"), act.dtype)
                       if self.point == "attn.scores" else mx.zeros_like(act))
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
            out = mx.where(mask, new, act)
            if (self.point == "attn.weights" and op == "zero" and self.renormalize):
                # A cut edge's mass has to go somewhere: the rows that
                # lost one are renormalised, the rest are left untouched
                # rather than divided by a rounded 1.
                f = out.astype(mx.float32)
                total = mx.sum(f, axis=feat_axis, keepdims=True)
                touched = mx.sum(mask.astype(mx.float32), axis=feat_axis, keepdims=True) > 0
                out = mx.where(touched, (f / mx.maximum(total, 1e-9)).astype(act.dtype), out)
            return out

        return fn
