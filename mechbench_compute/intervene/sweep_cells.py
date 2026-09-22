from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.intervene.coerce_axis_coord import coerce_axis_coord
from mechbench_compute.intervene.cell import Cell
from mechbench_compute.intervene.constants import SWEEP_AXES
from mechbench_compute.intervene.spec_error import SpecError


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
            coords = {SWEEP_AXES[axis]: coerce_axis_coord(value) for axis, value in combo}
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
