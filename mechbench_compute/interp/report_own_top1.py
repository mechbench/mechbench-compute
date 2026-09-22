from __future__ import annotations

from typing import Any

import numpy as np

from mechbench_compute import shapes as S


def report_own_top1(model, tok: int, lp: np.ndarray | None) -> dict[str, Any]:
    """`{"own_top1": token}` when the model's own top-1 under `lp` is not
    the target being measured — empty when it is, or when there is no
    baseline to ask (task 000597).

    A tracked target that is not the model's answer is often the point
    (measure THIS token's dependence), so this refuses nothing. But it
    is also how a mis-tokenized target hides: `tracked` says to include
    the leading space, which is right for a raw completion and wrong
    after a chat template's assistant prefix, where `" Paris"` and
    `"Paris"` are different tokens. A reader who sees the model's own
    answer beside the target knows at a glance which case they are in.
    """
    if lp is None:
        return {}
    top = int(np.argmax(lp))
    if top == tok:
        return {}
    return {"own_top1": {**S.token(model.tokenizer, top),
                         "logp": round(float(lp[top]), 4)}}
