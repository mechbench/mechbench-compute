from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.interp.collect_tracked_ids import collect_tracked_ids


def resolve_target(model, record: Mapping[str, Any], params: Mapping[str, Any],
                   lp: np.ndarray | None) -> tuple[int, dict[str, int]]:
    tracked = collect_tracked_ids(model, record, tracked=params.get("tracked"))
    if tracked:
        return next(iter(tracked.values())), tracked
    if lp is None:
        raise ValueError(
            f"record {record.get('id')!r}: no `tracked` token and no "
            "baseline to take the model's top-1 from")
    return int(np.argmax(lp)), tracked
