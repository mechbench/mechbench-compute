from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.intervene.cell import Cell
from mechbench_compute.intervene.constants import SWEEP_AXES


def sweep_as_run(sweep: Mapping[str, Any], cells: Sequence[Cell]) -> dict[str, Any]:
    """The header's record of a sweep: every axis and the values it took,
    strength including the control's 0 when one was added."""
    out: dict[str, Any] = {"strength": list(dict.fromkeys(c.factor for c in cells))}
    for axis in SWEEP_AXES:
        if axis != "strength" and axis in sweep:
            out[axis] = list(sweep[axis])
    return out
