from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute.directions.collect_models import collect_models


def resolve_space(vectors: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                  point: str | None) -> dict[str, Any]:
    """The rows' shared space, with the point overridden when asked.

    Rows from more than one model — a base and an adapted capture in one
    union — give a direction that lives in neither: its `model` is None
    (the residual basis they share), and the derivation names them all,
    so two such axes compare and the models are still on record."""
    sp = S.space_of(rows[0], header=vectors)
    if len(collect_models(vectors, rows)) > 1:
        sp["model"] = None
    if point:
        sp["point"] = P.normalize(str(point))
    sp["head"] = None
    return sp
