from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.intervene.cell import Cell
from mechbench_compute.intervene.scale_specs import scale_specs
from mechbench_compute.intervene.spec import Spec


class Compiled:
    def __init__(self, specs: list[Spec], weight_items: list[dict[str, Any]],
                 filled: list[dict[str, Any]], *,
                 activation_items: Sequence[Mapping[str, Any]] = (),
                 n_layers: int = 0, seed: int = 0) -> None:
        self.specs, self.weight_items, self.filled = specs, weight_items, filled
        self.activation_items = [dict(it) for it in activation_items]
        self.n_layers, self.seed = int(n_layers), int(seed)

    def at(self, cell: Cell) -> list[Spec]:
        if not cell.overrides:
            return scale_specs(self.specs, cell.factor)
        out: list[Spec] = []
        for item, spec in zip(self.activation_items, self.specs, strict=True):
            axes = item.get("sweep_over")
            applies = {a: v for a, v in cell.overrides.items()
                       if axes is None or a in axes}
            if not applies:
                out.extend(scale_specs([spec], cell.factor))
                continue
            out.extend(scale_specs(
                [Spec({**item, **applies}, n_layers=self.n_layers, seed=self.seed)],
                cell.factor))
        return out
