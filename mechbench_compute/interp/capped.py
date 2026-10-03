from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mechbench_compute.arrays import read_f64
from mechbench_compute.interp.answer import Answer

SATURATED_AT = 0.8


@dataclass(frozen=True)
class Capped:
    softcap: float
    logits: np.ndarray
    precap: np.ndarray

    def read(self, answer: Answer, lp: np.ndarray) -> dict[str, Any]:
        tid = answer.anchored(lp).preferred
        precap = float(self.precap[tid])
        return {"logit": round(float(self.logits[tid]), 4),
                "precap_logit": round(precap, 4),
                "saturated": abs(precap) > SATURATED_AT * self.softcap}


def read_capped(model, logits: Any, normed: Any) -> Capped:
    unembed = model.architecture.attribution_unembed(model._model)
    return Capped(float(unembed.softcap), read_f64(logits).reshape(-1),
                  read_f64(unembed.project(normed[:, -1:, :])).reshape(-1))
