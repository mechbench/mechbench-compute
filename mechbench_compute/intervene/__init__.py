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
    return [c.factor for c in sweep_cells(params)]
