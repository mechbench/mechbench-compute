from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.blocks.grid_rows import grid_rows


def cell_rows(recs: Sequence[Any]) -> list[Any]:
    """Records as rows: a grid's cells expanded (`grid_rows`), any other
    record as it is. The one reader for an op that takes a record
    stream and may be handed a trace — `records/summarize` over a patch
    trace with `value: share, by: [position]` is the strip of what each
    token's best cell recovers, without a plot in between."""
    out: list[Any] = []
    for r in recs:
        cells = grid_rows(r)
        if cells is None:
            out.append(r)
        else:
            out.extend(cells)
    return out
