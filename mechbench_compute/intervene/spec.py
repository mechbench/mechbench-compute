from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import directions as dirs
from mechbench_compute import positions as POS
from mechbench_compute._mlx import mx
from mechbench_compute.intervene.bind_constants import bind_constants, check_constants
from mechbench_compute.intervene.coerce_int_list import coerce_int_list
from mechbench_compute.intervene.build_rows_matrix import build_rows_matrix
from mechbench_compute.intervene.compile_operator import compile_operator
from mechbench_compute.intervene.operator_refused import OperatorRefused
from mechbench_compute.intervene.read_mask import read_mask
from mechbench_compute.intervene.spec_error import SpecError
from mechbench_compute.points import LAYOUT as _LAYOUT

OPS = ("zero", "mean", "resample", "patch", "add", "scale", "clamp",
       "project_out", "rotate")

BESIDE_F = {"op": "an item applies a fixed `op` or a function `f`, not both",
            "neurons": "an operator's dimensions are its `mask`",
            "direction": "an operator's direction is its `mask`",
            "direction2": "`direction2` is `rotate`'s", "row": "`row` is `patch`'s",
            "from": "`from` is `patch`'s",
            "pattern": "an operator acts at positions, not on an attention edge"}


_GLOBAL_POINTS = frozenset({"embed", "final_norm", "logits"})


_SAME = object()


class Spec:
    def __init__(self, item: Mapping[str, Any], *, n_layers: int, seed: int) -> None:
        point = str(item.get("point", "resid_post"))
        if point not in _LAYOUT:
            raise SpecError(f"unknown or unsupported point {point!r}; "
                            f"one of {sorted(_LAYOUT)}")
        f = item.get("f")
        self.operator = self.mask = None
        self.constants: dict[str, Any] = {}
        if f is None:
            stray = [k for k in ("mask", "constants") if item.get(k) is not None]
            if stray:
                raise OperatorRefused("OPERATOR_FIELDS", f"`{stray[0]}` belongs to an operator: give "
                                      "`f`, the function it applies", construct=stray[0])
        else:
            beside = [k for k in BESIDE_F if item.get(k) is not None]
            if beside:
                raise OperatorRefused("OPERATOR_FIELDS", f"{BESIDE_F[beside[0]]}: leave `{beside[0]}` "
                                      "out of an item that gives `f`", construct=beside[0])
            if isinstance(f, bool) or not isinstance(f, (str, int, float)):
                raise OperatorRefused("OPERATOR_SYNTAX", "`f` is an expression, written as a string",
                                      construct="f")
            self.operator = compile_operator(str(f))
            self.constants = dict(item.get("constants") or {})
            check_constants(self.constants, self.operator.names)
            self.mask = read_mask(item.get("mask"), invert=bool(item.get("except", False)))
        op = "f" if f is not None else str(item.get("op", "zero"))
        if f is None and op not in OPS:
            raise SpecError(f"unknown op {op!r}; one of {OPS}")
        self.point, self.op = point, op
        layers = item.get("layers", "all")
        if point in _GLOBAL_POINTS:
            self.layers: list[int | None] = [None]
        elif layers == "all":
            self.layers = list(range(n_layers))
        else:
            self.layers = coerce_int_list(layers)
            for layer in self.layers:
                if not 0 <= int(layer) < n_layers:
                    raise SpecError(f"layer {layer} out of range (n_layers={n_layers})")
        self.positions = item.get("positions", "last")
        self.heads = None if item.get("heads") is None else coerce_int_list(item["heads"])
        self.neurons = None if item.get("neurons") is None else coerce_int_list(item["neurons"])
        self.excepted = bool(item.get("except", False))
        if self.excepted:
            named = [k for k in ("layers", "heads", "neurons") if item.get(k) is not None]
            if self.mask is not None and self.mask.dims is not None:
                named.append("mask")
            if point in _GLOBAL_POINTS:
                raise SpecError(
                    f"`except` inverts a set of layers, heads or neurons; point "
                    f"{point!r} has none — it occurs once per forward pass")
            if not named:
                raise SpecError(
                    "`except` needs a set to invert: name `layers`, `heads` or "
                    "`neurons` on the item, or an operator's dimensions as its `mask`")
            if item.get("layers") is not None and layers != "all":
                keep = {int(x) for x in self.layers}
                self.layers = [i for i in range(n_layers) if i not in keep]
        self.patch_from = item.get("from")
        if self.patch_from is not None:
            if op != "patch":
                raise SpecError("`from` names where a `patch` reads; this item's op is "
                                f"{op!r}")
            if not isinstance(self.patch_from, Mapping):
                raise SpecError("`from` is an object: {layer, point}")
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
        self.direction = (dirs.coerce_array(item["direction"])
                          if item.get("direction") is not None else None)
        self.direction2 = (dirs.coerce_array(item["direction2"])
                           if item.get("direction2") is not None else None)
        self.source = item.get("source")
        self.row = item.get("row")
        cond = item.get("condition")
        self.condition = None
        if cond:
            self.condition = (dirs.coerce_array(cond["direction"]),
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
        bind = [k for k, v in self.constants.items() if isinstance(v, Mapping)]
        if bind and self.source is None:
            raise OperatorRefused(
                "CONSTANT_INVALID", f"constant `{bind[0]}` binds from `source`, and the item has no "
                "source and none arrived on the node's `source` port", construct=bind[0])
        self._rng = np.random.default_rng(self.seed)

    def hook_names(self) -> list[str]:
        return [self.point if layer is None else f"blocks.{layer}.{self.point}"
                for layer in self.layers]

    def build(self, layer: int | None, tokens: Sequence[str],
              record: Mapping[str, Any] | None = None,
              prompt_len: int | None = None, growing: bool = False,
              on_select: Callable[[list[int]], None] | None = None) -> Callable:
        pos_axis, head_axis, feat_axis = _LAYOUT[self.point]
        fixed_prompt_len = len(tokens) if prompt_len is None else int(prompt_len)
        absent = "none" if growing else "error"
        positions = self.positions
        heads, neurons = self.heads, self.neurons
        op, strength = self.op, self.strength
        d = None if self.direction is None else mx.array(self.direction)
        cond = self.condition
        rows = None
        if self.source is not None:
            src_layer, src_point = layer, self.point
            if self.patch_from is not None:
                src_layer = self.patch_from.get("layer", layer)
                src_point = self.patch_from.get("point", self.point)
            try:
                rows = build_rows_matrix(self.source, src_layer, src_point)
            except SpecError:
                rows = build_rows_matrix(self.source, src_layer)
        rng = self._rng

        def _positions(L: int, offset: int, selector: Any = _SAME) -> list[int]:
            n = offset + L
            want = positions if selector is _SAME else selector
            try:
                sel = POS.resolve(want, n, tokens=list(tokens[:n]), record=record,
                                  prompt_len=min(fixed_prompt_len, n), absent=absent)
            except ValueError as e:
                raise SpecError(str(e)) from None
            return [p - offset for p in sel if offset <= p < n]

        hook = self.point if layer is None else f"blocks.{layer}.{self.point}"
        bound: dict[int, dict[str, Any]] = {}

        def fn(act: mx.array, info) -> mx.array:
            shape = act.shape
            nd = len(shape)
            L = shape[pos_axis]
            offset = int(getattr(info, "offset", 0) or 0)
            if self.mask is not None:
                self.mask.check(shape[feat_axis], hook)
            sel_pos = _positions(L, offset)
            if not sel_pos:
                return act
            if on_select is not None:
                on_select([p + offset for p in sel_pos])

            def axis_mask(axis: int, idx: Sequence[int], invert: bool = False) -> mx.array:
                m = np.zeros(shape[axis], dtype=bool)
                m[list(idx)] = True
                if invert:
                    m = ~m
                view = [1] * nd
                view[axis] = shape[axis]
                return mx.array(m).reshape(view)

            if self.pattern is not None:
                to_sel = (sel_pos if "to" not in self.pattern
                          else _positions(L, offset, self.pattern["to"]))
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

            if self.operator is not None:
                return self._apply_operator(act, mask, rows, bound, hook)
            if op == "zero":
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
                f = out.astype(mx.float32)
                total = mx.sum(f, axis=feat_axis, keepdims=True)
                touched = mx.sum(mask.astype(mx.float32), axis=feat_axis, keepdims=True) > 0
                out = mx.where(touched, (f / mx.maximum(total, 1e-9)).astype(act.dtype), out)
            return out

        return fn

    def _apply_operator(self, act: mx.array, where: mx.array, rows: np.ndarray | None,
                        bound: dict[int, dict[str, Any]], hook: str) -> mx.array:
        d = act.shape[-1]
        if d not in bound:
            bound[d] = bind_constants(self.constants, self.mask, d, hook, rows)
        a = act.astype(mx.float32)
        x = self.mask.read(a)
        y = self.operator.evaluate({**bound[d], "x": x})
        if self.operator.undefinable:
            n = int(mx.sum(mx.logical_and(where, mx.logical_not(mx.isfinite(y)))).item())
            if n:
                raise OperatorRefused(
                    "OPERATOR_UNDEFINED", f"`{self.operator.canonical}` gave an undefined number (a "
                    "division by zero, the log or square root of a negative number, an overflow) at "
                    f"{n} coordinate{'' if n == 1 else 's'} of {hook}: the language's undefined number "
                    "is null, and an activation holds numbers", construct=self.operator.canonical)
        return mx.where(where, self.mask.write(a, x, y, self.strength).astype(act.dtype), act)
