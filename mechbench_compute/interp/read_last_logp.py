from __future__ import annotations

from typing import Any

import numpy as np

from mechbench_compute.arrays import read_logprobs


def read_last_logp(logits: Any) -> np.ndarray:
    return read_logprobs(logits[0, -1, :])
