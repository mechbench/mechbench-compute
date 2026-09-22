from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.directions.collect_models import collect_models


def build_model_provenance(vectors: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    models = collect_models(vectors, rows)
    return {"models": models} if len(models) > 1 else {}
