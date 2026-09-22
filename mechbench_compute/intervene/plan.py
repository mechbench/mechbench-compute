from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.intervene.cell import Cell
from mechbench_compute.intervene.compile import compile
from mechbench_compute.intervene.compiled import Compiled
from mechbench_compute.intervene.spec import Spec
from mechbench_compute.intervene.spec_intervention import SpecIntervention
from mechbench_compute.intervene.read_spec_items import read_spec_items
from mechbench_compute.intervene.sweep_as_run import sweep_as_run
from mechbench_compute.intervene.sweep_cells import sweep_cells
from mechbench_compute.intervene.serialize_spec import serialize_spec


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
        return {"spec": serialize_spec(self.compiled.filled),
                "weights": [dict(it) for it in self.weight_items] or None,
                "sweep": sweep_as_run(self._sweep, self.cells)}


def plan(model, params: Mapping[str, Any], inputs: Mapping[str, Any] | None) -> Plan | None:
    """A text op's intervention, planned: None when the node names none."""
    items = read_spec_items(params.get("spec"), inputs)
    if not items:
        return None
    compiled = compile(model, items, inputs=inputs, seed=int(params.get("seed", 0)))
    return Plan(compiled, sweep_cells(params), params.get("sweep") or {})
