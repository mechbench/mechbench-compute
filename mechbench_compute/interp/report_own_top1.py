from __future__ import annotations

from typing import Any

import numpy as np

from mechbench_compute import shapes as S


def report_own_top1(model, tok: int, lp: np.ndarray | None) -> dict[str, Any]:
    if lp is None:
        return {}
    top = int(np.argmax(lp))
    if top == tok:
        return {}
    return {"own_top1": {**S.token(model.tokenizer, top),
                         "logp": round(float(lp[top]), 4)}}
