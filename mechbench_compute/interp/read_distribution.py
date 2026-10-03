from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.interp.answer import Answer
from mechbench_compute.interp.capped import Capped


def read_distribution(tokenizer, lp: np.ndarray, *, top_k: int,
                      tracked: Mapping[str, Answer] | None = None,
                      capped: Capped | None = None) -> dict[str, Any]:
    out = S.distribution(lp, tokenizer, top_k=top_k)
    if tracked:
        out["tracked"] = {str(name): {**answer.entry(tokenizer, lp),
                                      **(capped.read(answer, lp) if capped is not None else {})}
                          for name, answer in tracked.items()}
    return out
