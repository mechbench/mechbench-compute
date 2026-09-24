from __future__ import annotations

from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.interp.answer import Answer


def report_own_top1(model, answer: Answer, lp: np.ndarray | None) -> dict[str, Any]:
    if lp is None:
        return {}
    top = int(np.argmax(lp))
    if top in answer.ids:
        return {}
    return {"own_top1": {**S.token(model.tokenizer, top),
                         "logp": round(float(lp[top]), 4)}}
