from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.directions.models_of import _models_of


def _model_provenance(vectors: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    models = _models_of(vectors, rows)
    return {"models": models} if len(models) > 1 else {}
