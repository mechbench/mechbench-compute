from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.interp.answer import Answer
from mechbench_compute.interp.read_last_logp import read_last_logp

METRICS = ("logprob", "prob", "logit", "entropy", "entropy_outcomes", "mass_outcomes")

OUTCOME_METRICS = ("entropy_outcomes", "mass_outcomes")

METRIC_DOC = (
    "What is read at the decision position: the target's `logprob`, `prob` or `logit`; `entropy`, the whole "
    "next-token distribution's entropy in bits; `entropy_outcomes`, the entropy in bits of the distribution "
    "renormalised over the record's `outcomes`, so six faces equally likely read 2.585 bits whatever the mass "
    "on them; or `mass_outcomes`, the probability on those outcomes together. `entropy` misleads when an "
    "intervention breaks the model: a broken model spreads its mass over the whole vocabulary and reads as "
    "flat, which is also what a flattening behaviour reads as. `entropy_outcomes` asks only how the mass is "
    "shared among the answers the record allows; read `mass_outcomes` beside it, since a share of almost "
    "nothing is still a share. An outcome is matched as a `tracked` answer is, by its first token written "
    "with and without a leading space, keeping the spellings whose token is the whole outcome; a record "
    "without `outcomes`, or with an outcome the tokenizer splits, is refused by its id.")


def read_metric(answer: Answer, metric: str, logits: mx.array,
                outcomes: Sequence[Answer] | None = None) -> float:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}: one of {', '.join(METRICS)}")
    lp = read_last_logp(logits)
    if metric == "entropy":
        p = np.exp(np.asarray(lp, dtype=np.float64))
        nz = p[p > 0]
        return float(-(nz * np.log2(nz)).sum())
    if metric in OUTCOME_METRICS:
        if not outcomes:
            raise ValueError(f"metric {metric!r} reads a record's `outcomes`, and none were given")
        p = np.array([o.p(lp) for o in outcomes], dtype=np.float64)
        mass = float(p.sum())
        if metric == "mass_outcomes":
            return mass
        q = p[p > 0] / mass
        return float(-(q * np.log2(q)).sum())
    if metric == "logit":
        row = logits[0, -1, :].astype(mx.float32)
        mx.eval(row)
        return answer.read(metric, lp, np.array(row))
    return answer.read(metric, lp)
