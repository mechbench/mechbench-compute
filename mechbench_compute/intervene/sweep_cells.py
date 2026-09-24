from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.intervene.coerce_axis_coord import coerce_axis_coord
from mechbench_compute.intervene.cell import Cell
from mechbench_compute.intervene.constants import SWEEP_AXES
from mechbench_compute.intervene.spec_error import SpecError


def sweep_cells(params: Mapping[str, Any]) -> list[Cell]:
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
        cells = [Cell(0.0, {}, {}, "control" if others else None), *cells]
    return cells
