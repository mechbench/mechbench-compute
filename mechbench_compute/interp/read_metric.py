from __future__ import annotations

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.interp.answer import Answer
from mechbench_compute.interp.read_last_logp import read_last_logp

METRICS = ("logprob", "prob", "logit", "entropy")


def read_metric(answer: Answer, metric: str, logits: mx.array) -> float:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}: one of {', '.join(METRICS)}")
    lp = read_last_logp(logits)
    if metric == "entropy":
        p = np.exp(np.asarray(lp, dtype=np.float64))
        nz = p[p > 0]
        return float(-(nz * np.log2(nz)).sum())
    if metric == "logit":
        row = logits[0, -1, :].astype(mx.float32)
        mx.eval(row)
        return answer.read(metric, lp, np.array(row))
    return answer.read(metric, lp)
