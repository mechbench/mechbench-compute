from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.intervene.cell import Cell
from mechbench_compute.intervene.compiled import Compiled
from mechbench_compute.intervene.spec_error import SpecError


def drop_unmatched(compiled: Compiled, records: Sequence[Mapping[str, Any]],
                   cells: Sequence[Cell]) -> tuple[list[Mapping[str, Any]], list[Any]]:
    if not any(spec.match is not None for spec in compiled.specs):
        return list(records), []
    runs = [compiled.at(cell) for cell in cells if cell.factor != 0.0]
    kept, unmatched = [], []
    for record in records:
        if any(spec.unmatched_layers(record) for specs in runs for spec in specs):
            unmatched.append(record.get("id"))
        else:
            kept.append(record)
    if not kept:
        raise SpecError("`row.match` found no `source` row for any record: "
                        f"{', '.join(map(str, unmatched))}")
    return kept, unmatched
