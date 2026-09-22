"""The declarative `intervene` block.

Intervention is a product of two small sets. POINTS: any hook point of
the forward pass. OPERATIONS: zero, mean, resample, patch, add, scale,
clamp, project_out, rotate — over sets of positions, heads and
neurons, optionally conditioned on a direction's projection or on
token identity. One spec list, applied together in one forward, then
a readout (the next-token distribution, or captures). Ablation and
steering are special cases of this grammar.

A spec item::

    {"point": "resid_post", "layers": [14] | "all",
     "positions": "last" | "all" | [3, 5] | {"tokens": ["lighthouse"]} | {"range": [2, 6]},
     "heads": [0, 3] | null, "neurons": [17, 902] | null,
     "op": "add", "strength": 4.0,
     "direction": <direction object>, "direction2": <direction> (rotate),
     "source": <residual_vectors record> (mean / resample / patch), "row": {...},
     "condition": {"direction": <direction>, "threshold": 0.0, "above": true}}

Everything here is deterministic given the spec and the records
(`resample` draws from `seed`), so the block is `reproducible`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.intervene.compile import compile  # noqa: F401
from mechbench_compute.intervene.edit_weights import edit_weights  # noqa: F401
from mechbench_compute.intervene.plan import plan  # noqa: F401
from mechbench_compute.intervene.scale_specs import scale_specs  # noqa: F401
from mechbench_compute.intervene.spec import Spec  # noqa: F401
from mechbench_compute.intervene.spec_error import SpecError  # noqa: F401
from mechbench_compute.intervene.spec_intervention import SpecIntervention  # noqa: F401
from mechbench_compute.intervene.sweep_cells import sweep_cells  # noqa: F401


def sweep_factors(params: Mapping[str, Any]) -> list[float]:
    """The strengths a node's `sweep` runs — `sweep_cells` read by a
    caller that varies nothing else."""
    return [c.factor for c in sweep_cells(params)]

