from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TorchNorm:
    module: Any
    gain_offset: float

    @property
    def weight(self) -> Any:
        return self.module.weight

    @property
    def eps(self) -> float:
        return float(getattr(self.module, "eps", getattr(self.module, "variance_epsilon", 1e-6)))

    def __call__(self, x: Any) -> Any:
        return self.module(x)
