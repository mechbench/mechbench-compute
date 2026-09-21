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
from mechbench_compute.intervene.as_list import _as_list  # noqa: F401
from mechbench_compute.intervene.axis_coord import _axis_coord  # noqa: F401
from mechbench_compute.intervene.cell import Cell  # noqa: F401
from mechbench_compute.intervene.compile import compile  # noqa: F401
from mechbench_compute.intervene.compiled import Compiled  # noqa: F401
from mechbench_compute.intervene.constants import SWEEP_AXES  # noqa: F401
from mechbench_compute.intervene.edited import edited  # noqa: F401
from mechbench_compute.intervene.plan import Plan, plan  # noqa: F401
from mechbench_compute.intervene.rows_matrix import _rows_matrix  # noqa: F401
from mechbench_compute.intervene.scaled import scaled  # noqa: F401
from mechbench_compute.intervene.source_items import _source_items  # noqa: F401
from mechbench_compute.intervene.spec import OPS, Spec, _GLOBAL_POINTS, _SAME  # noqa: F401
from mechbench_compute.intervene.spec_error import SpecError  # noqa: F401
from mechbench_compute.intervene.spec_intervention import SpecIntervention  # noqa: F401
from mechbench_compute.intervene.spec_items import spec_items  # noqa: F401
from mechbench_compute.intervene.sweep_as_run import sweep_as_run  # noqa: F401
from mechbench_compute.intervene.sweep_cells import sweep_cells  # noqa: F401
from mechbench_compute.intervene.wire_spec import _wire_spec  # noqa: F401


# --- parsing -----------------------------------------------------------------------


def _hook_space(name: str) -> tuple[int | None, str]:
    """`blocks.14.resid_post` → (14, "resid_post"); a whole-model point
    → (None, name)."""
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) >= 3 and parts[1].isdigit():
        return int(parts[1]), ".".join(parts[2:])
    return None, name


def sweep_factors(params: Mapping[str, Any]) -> list[float]:
    """The strengths a node's `sweep` runs — `sweep_cells` read by a
    caller that varies nothing else."""
    return [c.factor for c in sweep_cells(params)]


# --- the block ---------------------------------------------------------------------


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
