from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.directions.constants import KIND


def make(vector: Any, sp: Mapping[str, Any], *, method: str,
         sources: Sequence[str] = (), labels: Mapping[str, Any] | None = None,
         extra: Mapping[str, Any] | None = None, unit: bool = True) -> dict[str, Any]:
    """A direction in space `sp`. `labels` is the grouping it was built
    from (`{axis, positive, negative}`); `extra` rides in the derivation."""
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if unit:
        if norm == 0.0:
            raise ValueError("a zero vector cannot be a direction")
        v = v / norm
    prov: dict[str, Any] = {"method": method, "sources": list(sources),
                            "model": sp.get("model")}
    if labels:
        prov.update({k: v_ for k, v_ in labels.items() if v_ is not None})
    if extra:
        prov.update(dict(extra))
    # `derivation`, not `provenance`: the latter is the Emitted envelope's
    # field, and bench.emit refuses to wrap a payload that carries one.
    item = S.vector(v, sp)
    item["norm"] = round(norm, 6)
    item["unit"] = bool(unit)
    item["derivation"] = prov
    return {"kind": KIND, **item}
