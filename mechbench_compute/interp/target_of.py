from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.interp.tracked_ids import _tracked_ids


def _target_of(model, record: Mapping[str, Any], params: Mapping[str, Any],
               lp: np.ndarray | None) -> tuple[int, dict[str, int]]:
    """(the target token, every tracked token): the first `tracked`
    entry is the target; with none named, the model's own top-1 under
    `lp` is. Returns the tracked map too, so a readout can report every
    named token."""
    tracked = _tracked_ids(model, record, tracked=params.get("tracked"))
    if tracked:
        return next(iter(tracked.values())), tracked
    if lp is None:
        raise ValueError(
            f"record {record.get('id')!r}: no `tracked` token and no "
            "baseline to take the model's top-1 from")
    return int(np.argmax(lp)), tracked
