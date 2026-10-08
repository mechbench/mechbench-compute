from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import positions as POS
from mechbench_compute.intervene.cell import Cell
from mechbench_compute.intervene.compile import compile
from mechbench_compute.intervene.compiled import Compiled
from mechbench_compute.intervene.spec import Spec
from mechbench_compute.intervene.spec_error import SpecError
from mechbench_compute.intervene.spec_intervention import SpecIntervention
from mechbench_compute.intervene.read_spec_items import read_spec_items
from mechbench_compute.intervene.sweep_as_run import sweep_as_run
from mechbench_compute.intervene.sweep_cells import sweep_cells
from mechbench_compute.intervene.serialize_spec import serialize_spec


class Plan:
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
        if cell.factor == 0.0 or not self.specs:
            return []
        return [SpecIntervention(self.compiled.at(cell), tokens, record, growing=True)]

    def refuse_steps_past(self, max_tokens: int) -> None:
        selectors = []
        for it in self.compiled.activation_items:
            selectors.append(it.get("positions"))
            pattern = it.get("pattern")
            if isinstance(pattern, Mapping):
                selectors += [pattern.get("from"), pattern.get("to")]
        selectors += list(self._sweep.get("positions") or [])
        for sel in selectors:
            k = POS.furthest_step(sel)
            if k is not None and k >= int(max_tokens):
                raise SpecError(
                    f"step {k} is past the end: this node writes at most {int(max_tokens)} "
                    f"token{'' if int(max_tokens) == 1 else 's'} (steps 0 to {int(max_tokens) - 1}); "
                    "raise `max_tokens` or name an earlier step")

    def header(self) -> dict[str, Any]:
        return {"spec": serialize_spec(self.compiled.filled),
                "weights": [dict(it) for it in self.weight_items] or None,
                "sweep": sweep_as_run(self._sweep, self.cells)}


def read_steps(lives: Sequence[Sequence[Any]], max_tokens: int) -> list[int]:
    seen: set[int] = set()
    for live in lives:
        for iv in live or ():
            seen.update(k for k in getattr(iv, "steps", ()) if k < int(max_tokens))
    return sorted(seen)


def plan(model, params: Mapping[str, Any], inputs: Mapping[str, Any] | None) -> Plan | None:
    items = read_spec_items(params.get("spec"), inputs)
    if not items:
        return None
    compiled = compile(model, items, inputs=inputs, seed=int(params.get("seed", 0)))
    return Plan(compiled, sweep_cells(params), params.get("sweep") or {})

