from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.interp.answer import Answer, make_answer
from mechbench_compute.interp.collect_tracked_answers import collect_tracked_answers


def resolve_target(model, record: Mapping[str, Any], params: Mapping[str, Any],
                   lp: np.ndarray | None) -> tuple[Answer, dict[str, Answer]]:
    tracked = {name: answer.anchored(lp) for name, answer in
               collect_tracked_answers(model, record, tracked=params.get("tracked")).items()}
    if tracked:
        return next(iter(tracked.values())), tracked
    if lp is None:
        raise ValueError(
            f"record {record.get('id')!r}: no `tracked` token and no "
            "baseline to take the model's top-1 from")
    return make_answer([int(np.argmax(lp))]), tracked
