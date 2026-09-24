from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.blocks.expand_grid import expand_grid


def expand_cells(recs: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for r in recs:
        cells = expand_grid(r)
        if cells is None:
            out.append(r)
        else:
            out.extend(cells)
    return out
