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

import contextlib
import json

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

#: "the selector this item already names" — a sentinel, because None is
#: a selector (`positions: null` is `"last"`).
_SAME = object()


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


class Compiled:
    """A spec list parsed against a model: the activation items as
    `Spec`s, the weight items as written, and the items as filled from
    the ports — what lineage records. `at(cell)` re-parses the items
    with a sweep cell's fields overridden (000602)."""

    def __init__(self, specs: list[Spec], weight_items: list[dict[str, Any]],
                 filled: list[dict[str, Any]], *,
                 activation_items: Sequence[Mapping[str, Any]] = (),
                 n_layers: int = 0, seed: int = 0) -> None:
        self.specs, self.weight_items, self.filled = specs, weight_items, filled
        self.activation_items = [dict(it) for it in activation_items]
        self.n_layers, self.seed = int(n_layers), int(seed)

    def at(self, cell: Cell) -> list[Spec]:
        """The specs this cell runs: each item with the cell's fields
        set — unless the item's `sweep_over` names a smaller set of axes
        — and every strength multiplied by the cell's factor."""
        if not cell.overrides:
            return scaled(self.specs, cell.factor)
        out: list[Spec] = []
        for item, spec in zip(self.activation_items, self.specs, strict=True):
            axes = item.get("sweep_over")
            applies = {a: v for a, v in cell.overrides.items()
                       if axes is None or a in axes}
            if not applies:
                out.extend(scaled([spec], cell.factor))
                continue
            out.extend(scaled(
                [Spec({**item, **applies}, n_layers=self.n_layers, seed=self.seed)],
                cell.factor))
        return out


def compile(model, items: Sequence[Mapping[str, Any]], *,
            inputs: Mapping[str, Any] | None = None, seed: int = 0) -> Compiled:
    """Parse spec items for `model`. Objects may arrive by edge: a
    `direction` / `source` port in `inputs` fills any item that names none
    of its own. An item that names a `parameter` edits a WEIGHT, not an
    activation (task 000457): its scope is the node rather than the
    forward pass — the tensor is changed, every record runs against the
    changed model, and the original is reinstalled afterwards. The two
    kinds compose, so they are separated here and applied in their own
    scopes. Shared by `intervene/apply` and by the text ops that take an
    intervention (000601)."""
    inputs = inputs or {}
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
    weight_items = [it for it in filled if it.get("parameter") is not None]
    activation_items = [it for it in filled if it.get("parameter") is None]
    for it in activation_items:
        over = it.get("sweep_over")
        if over is None:
            continue
        if not isinstance(over, (list, tuple)) or any(a not in SWEEP_AXES for a in over):
            raise SpecError(
                f"`sweep_over` names sweep axes ({', '.join(SWEEP_AXES)}), "
                f"not {over!r}")
    specs = [Spec(it, n_layers=model.arch.n_layers, seed=int(seed)) for it in activation_items]
    if not specs and not weight_items:
        raise SpecError("an intervention needs a non-empty spec list")
    return Compiled(specs, weight_items, filled, activation_items=activation_items,
                    n_layers=model.arch.n_layers, seed=int(seed))


def spec_items(inline: Any, inputs: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The spec items a node was given: its inline list when present,
    else the `items` of an `intervene/spec` object on its `intervention`
    port; [] when neither (000601)."""
    if inline:
        return [dict(it) for it in inline]
    port = (inputs or {}).get("intervention")
    if port is None:
        return []
    if not isinstance(port, Mapping) or not isinstance(port.get("items"), list):
        raise SpecError("the `intervention` port takes an intervene/spec: "
                        "an object with `items`, the spec items")
    return [dict(it) for it in port["items"]]


class Plan:
    """What a text op runs under an intervention: the compiled specs,
    the sweep's cells, and a way to make one record's live intervention
    per cell. None of it when the node has no intervention."""

    def __init__(self, compiled: Compiled, cells: list[Cell],
                 sweep: Mapping[str, Any] | None = None) -> None:
        self.compiled, self.cells = compiled, cells
        self._sweep = dict(sweep or {})

    @property
    def factors(self) -> list[float]:
        return [c.factor for c in self.cells]

    @property
    def specs(self) -> list[Spec]:
        return self.compiled.specs

    @property
    def weight_items(self) -> list[dict[str, Any]]:
        return self.compiled.weight_items

    def live(self, cell: Cell, tokens: Sequence[str],
             record: Mapping[str, Any] | None = None) -> list[SpecIntervention]:
        """The interventions for one record in one cell: none in the
        control, else the cell's specs, over a token list this record's
        decoder grows."""
        if cell.factor == 0.0 or not self.specs:
            return []
        return [SpecIntervention(self.compiled.at(cell), tokens, record, growing=True)]

    def header(self) -> dict[str, Any]:
        """What the result records: the items as run, the weight edits,
        the sweep as run."""
        return {"spec": _wire_spec(self.compiled.filled),
                "weights": [dict(it) for it in self.weight_items] or None,
                "sweep": sweep_as_run(self._sweep, self.cells)}


def sweep_as_run(sweep: Mapping[str, Any], cells: Sequence[Cell]) -> dict[str, Any]:
    """The header's record of a sweep: every axis and the values it took,
    strength including the control's 0 when one was added."""
    out: dict[str, Any] = {"strength": list(dict.fromkeys(c.factor for c in cells))}
    for axis in SWEEP_AXES:
        if axis != "strength" and axis in sweep:
            out[axis] = list(sweep[axis])
    return out


def plan(model, params: Mapping[str, Any], inputs: Mapping[str, Any] | None) -> Plan | None:
    """A text op's intervention, planned: None when the node names none."""
    items = spec_items(params.get("spec"), inputs)
    if not items:
        return None
    compiled = compile(model, items, inputs=inputs, seed=int(params.get("seed", 0)))
    return Plan(compiled, sweep_cells(params), params.get("sweep") or {})


#: The spec fields a sweep may vary, in the order the product runs them,
#: and the coordinate each one becomes on the result (000602). `strength`
#: is the outermost, because a weight edit is applied once per strength
#: and every cell under it runs against the edited model.
SWEEP_AXES: dict[str, str] = {"strength": "factor", "layers": "layer",
                              "heads": "head", "positions": "position",
                              "neurons": "neuron"}


def _axis_coord(value: Any) -> Any:
    """What a swept value reads as on the result: a scalar as itself, a
    one-element list as its element (so `layers: [[0],[1]]` gives 0 and
    1), anything else as a compact label — a coordinate is grouped on,
    so it must be a value a group key can hold."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _axis_coord(value[0])
        return "+".join(str(_axis_coord(v)) for v in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


class Cell:
    """One cell of a sweep: the strength factor, the spec fields it
    overrides, and the coordinates it puts on every row it produces."""

    def __init__(self, factor: float, overrides: Mapping[str, Any],
                 coords: Mapping[str, Any], label: str | None) -> None:
        self.factor = float(factor)
        self.overrides = dict(overrides)
        #: The axis coordinates beside `factor` — `layer`, `head`, …
        self.coords = dict(coords)
        #: A one-word name for the cell, when the sweep runs axes beyond
        #: strength; None when it does not, so a strength-only sweep's
        #: rows are identified by `factor` exactly as they always were.
        self.label = label

    @property
    def key(self) -> str:
        return self.label or f"{self.factor}"

    @property
    def slug(self) -> str:
        """The cell in an id: `f1` for a strength-only sweep, as it has
        always been; `layer0`, `factor2-layer3` when other axes run."""
        if not self.label:
            return f"f{self.factor:g}"
        return self.label.replace("=", "").replace("/", "-")

    @property
    def axes(self) -> dict[str, Any]:
        """Every coordinate this cell puts on a row, `factor` included."""
        return {"factor": self.factor, **self.coords}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Cell(factor={self.factor}, overrides={self.overrides})"


def sweep_cells(params: Mapping[str, Any]) -> list[Cell]:
    """The cells a node's `sweep` runs: the cartesian product of its
    axes in `SWEEP_AXES` order, with the untouched-model control first
    unless `control` is false or a strength of 0 is already named.

    A sweep over `strength` alone is the sweep this op has always run —
    one cell per factor, `factor` the only coordinate. Any other axis
    (`layers`, `heads`, `positions`, `neurons`) sets that field on every
    spec item it applies to, and becomes a coordinate of its own, so a
    layer sweep is one node and one result where it used to be a
    `records/map` over a corpus of integers (000602).
    """
    sweep = params.get("sweep") or {}
    if not isinstance(sweep, Mapping):
        raise SpecError("`sweep` is an object of axes: "
                        f"{', '.join(SWEEP_AXES)}")
    unknown = [k for k in sweep if k not in SWEEP_AXES]
    if unknown:
        raise SpecError(
            f"`sweep` cannot vary {', '.join(sorted(unknown))}; the axes are "
            f"{', '.join(SWEEP_AXES)}")
    for axis, values in sweep.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise SpecError(f"`sweep.{axis}` is a non-empty list of values")
    strengths = [float(f) for f in sweep.get("strength", [1.0])]
    others = [(a, list(sweep[a])) for a in SWEEP_AXES if a != "strength" and a in sweep]

    cells: list[Cell] = []
    for factor in strengths:
        # itertools.product over the non-strength axes, in axis order.
        combos: list[list[tuple[str, Any]]] = [[]]
        for axis, values in others:
            combos = [[*c, (axis, v)] for c in combos for v in values]
        for combo in combos:
            overrides = {axis: value for axis, value in combo}
            coords = {SWEEP_AXES[axis]: _axis_coord(value) for axis, value in combo}
            label = None
            if others:
                parts = ([f"factor={factor:g}"] if "strength" in sweep else [])
                parts += [f"{SWEEP_AXES[a]}={coords[SWEEP_AXES[a]]}" for a, _ in combo]
                label = "/".join(parts)
            cells.append(Cell(factor, overrides, coords, label))
    if bool(params.get("control", True)) and not any(c.factor == 0.0 for c in cells):
        # The control is the model untouched, which is one run however
        # many axes the sweep has: an unintervened forward pass does not
        # depend on the layer the intervention would have named.
        cells = [Cell(0.0, {}, {}, "control" if others else None), *cells]
    return cells


def sweep_factors(params: Mapping[str, Any]) -> list[float]:
    """The strengths a node's `sweep` runs — `sweep_cells` read by a
    caller that varies nothing else."""
    return [c.factor for c in sweep_cells(params)]


class SpecIntervention:
    """An `Intervention` (as_hooks / as_captures) over a whole spec list
    for one record's tokens.

    `tokens` is the sequence the positions resolve against. A decoder
    that runs the sequence in chunks — the prompt, then one token per
    step — calls `on_token` with each token it produces, so a selector
    like `{"tokens": ["lighthouse"]}` or `"generated"` sees the words
    as they arrive (000601)."""

    def __init__(self, specs: Sequence[Spec], tokens: Sequence[str],
                 record: Mapping[str, Any] | None = None,
                 prompt_len: int | None = None, growing: bool = False) -> None:
        self.tokens: list[str] = list(tokens)
        self.prompt_len = len(self.tokens) if prompt_len is None else int(prompt_len)
        self._hooks: dict[str, Callable] = {}
        for spec in specs:
            for layer, name in zip(spec.layers, spec.hook_names(), strict=True):
                fn = spec.build(layer, self.tokens, record, prompt_len=self.prompt_len,
                                growing=growing)
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

    def on_token(self, token: str) -> None:
        """The decoder produced one more token: the sequence grew."""
        self.tokens.append(token)


def scaled(specs: Sequence[Spec], factor: float) -> list[Spec]:
    """The specs at one sweep factor: each strength multiplied. Factor 1
    is the specs themselves."""
    if factor == 1.0:
        return list(specs)
    out = []
    for spec in specs:
        s2 = Spec.__new__(Spec)
        s2.__dict__.update(spec.__dict__)
        s2.strength = spec.strength * factor
        out.append(s2)
    return out


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


@contextlib.contextmanager
def edited(model, weight_items: Sequence[Mapping[str, Any]], factor: float):
    """The model's weights edited by `weight_items` at `factor` for the
    block's duration, and put back whatever happens inside — a run that
    left a model edited would poison every later node in the job."""
    if not weight_items or factor == 0.0:
        yield
        return
    from mechbench_compute import weights as weights_mod

    handle = weights_mod.edit_parameters(model.lm, weight_items, factor)
    try:
        yield
    finally:
        weights_mod.restore_parameters(model.lm, handle)


def _order(records: Sequence[Mapping[str, Any]], cells: Sequence[Cell],
           weight_items: Sequence[Mapping[str, Any]], model):
    """(record, cell) pairs, with any weight edits in scope.

    Without weight items this is the loop it always was: record outer,
    cell inner. With them the strength goes outside, because a weight
    edit is applied once for every record that runs under it — and the
    edit is undone before the next strength, and before the generator
    returns, whatever happens in between. A run that left a model edited
    would poison every later node in the job, which is the failure this
    `finally` exists for. `sweep_cells` orders strength outermost, so
    the cells sharing one are contiguous and each edit is made once.
    """
    if not weight_items:
        for record in records:
            for cell in cells:
                yield record, cell
        return

    from mechbench_compute import weights as weights_mod

    for factor in dict.fromkeys(c.factor for c in cells):
        handle = ([] if factor == 0.0
                  else weights_mod.edit_parameters(model.lm, weight_items, factor))
        try:
            for record in records:
                for cell in cells:
                    if cell.factor == factor:
                        yield record, cell
        finally:
            weights_mod.restore_parameters(model.lm, handle)


def run(model, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
        on_item: Callable | None = None,
        on_start: Callable[[int], None] | None = None) -> dict[str, Any]:
    from mechbench_compute.distill import render
    from mechbench_compute.interp import _last_logp

    inputs = inputs or {}
    items = spec_items(params.get("spec"), inputs)
    if not items:
        raise SpecError("intervene needs a non-empty `spec` list, or an "
                        "intervene/spec on the `intervention` port")
    compiled = compile(model, items, inputs=inputs, seed=int(params.get("seed", 0)))
    specs, weight_items, filled = compiled.specs, compiled.weight_items, compiled.filled
    cells = sweep_cells(params)
    readout = dict(params.get("readout") or {"type": "decision"})
    rk = str(readout.get("type") or readout.get("kind") or "decision")
    if rk not in ("decision", "capture"):
        raise SpecError(f"unknown readout kind {rk!r}")
    top_k = int(readout.get("top_k", params.get("top_k", 5)))
    if not records:
        raise SpecError("intervene needs at least one record")
    if on_start:
        on_start(len(records) * len(cells))

    from mechbench_compute.interp import _tracked_ids
    from mechbench_compute.lexicon import kinds as K

    mid = S.model_id_of(model)
    rows: list[dict[str, Any]] = []
    # A weight edit is applied once per sweep factor, not once per
    # record: the tensor is the same for every record that runs under it,
    # and an SVD per record would be absurd. So the factor is the outer
    # loop when there are weight items, and the record loop is the same
    # body either way.
    for record, cell in _order(records, cells, weight_items, model):
            ids = render(model, record).array
            flat = [int(t) for t in np.array(ids).reshape(-1)]
            tokens = [model.tokenizer.decode([t]) for t in flat]
            tracked = _tracked_ids(model, record, tracked=params.get("tracked"))
            factor = cell.factor
            key = f"{record.get('id')}:{cell.key}"
            # The cell's axes ride on every row it produces: `factor` as
            # it always has, and a coordinate per other swept axis, so a
            # layer sweep groups on `layer` (000602).
            coords = {**record.get("coords", {}), **cell.coords}
            named = {"cell": cell.label} if cell.label else {}
            if factor == 0.0:
                ivs: list[Any] = []
            else:
                ivs = [SpecIntervention(compiled.at(cell), tokens, record)]
            if rk == "decision":
                res = model.run(ids, interventions=ivs)
                lp = _last_logp(res.logits)
                row: dict[str, Any] = {
                    "id": record.get("id"), "coords": dict(coords),
                    "factor": factor, **named,
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
                        id=record.get("id"), coords=dict(coords),
                        factor=factor, position=pidx, **named,
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
        sweep=sweep_as_run(params.get("sweep") or {}, cells),
        readout=rk,
        **({"model": mid, "position": str(readout.get("position", "last")),
            "points": [str(p) for p in readout.get("points", [])]} if rk == "capture" else {}),
        description=(
            f"{'; '.join(what)}. Factor 0 is the control; strengths scale "
            f"with the sweep factor."),
    )
