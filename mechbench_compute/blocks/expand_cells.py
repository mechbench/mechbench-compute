from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.blocks.expand_grid import expand_grid


def expand_cells(recs: Sequence[Any]) -> list[Any]:
    """Records as rows: a grid's cells expanded (`expand_grid`), any other
    record as it is. The one reader for an op that takes a record
    stream and may be handed a trace — `records/summarize` over a patch
    trace with `value: share, by: [position]` is the strip of what each
    token's best cell recovers, without a plot in between."""
    out: list[Any] = []
    for r in recs:
        cells = expand_grid(r)
        if cells is None:
            out.append(r)
        else:
            out.extend(cells)
    return out
