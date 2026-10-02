from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .arrays import read_softmax
from .cache import ActivationCache


@dataclass(frozen=True)
class RunResult:
    logits: Any
    cache: ActivationCache

    @property
    def last_logits(self) -> Any:
        return self.logits[0, -1, :]

    def top_k(self, tokenizer, k: int = 5) -> list[tuple[str, float]]:
        probs_np = self.last_probs()
        top_idx = np.argsort(-probs_np)[:k]
        return [
            (tokenizer.decode([int(i)]), float(probs_np[i])) for i in top_idx
        ]

    def top1(self, tokenizer) -> tuple[int, str, float]:
        probs = self.last_probs()
        top_id = int(np.argmax(probs))
        return top_id, tokenizer.decode([top_id]), float(probs[top_id])

    def last_probs(self) -> np.ndarray:
        return read_softmax(self.last_logits)
