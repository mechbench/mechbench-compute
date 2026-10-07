from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TorchOps:
    framework = "torch"
    float32 = torch.float32

    def __init__(self, device: Any) -> None:
        self.device = device

    def array(self, values: Any) -> Any:
        a = np.asarray(values)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        return torch.as_tensor(a, device=self.device)

    def cast(self, x: Any, dtype: Any) -> Any:
        return x.to(dtype)

    def size(self, x: Any) -> int:
        return int(x.numel())

    def sum(self, x: Any, axis: int) -> Any:
        return x.sum(dim=axis, keepdim=True)

    def full(self, shape: Any, value: float, dtype: Any) -> Any:
        return torch.full(tuple(shape), value, dtype=dtype, device=self.device)

    def zeros_like(self, x: Any) -> Any:
        return torch.zeros_like(x)

    def clip(self, x: Any, lo: float, hi: float) -> Any:
        return torch.clamp(x, lo, hi)

    def where(self, mask: Any, a: Any, b: Any) -> Any:
        return torch.where(mask, a, b)

    def maximum(self, x: Any, floor: float) -> Any:
        return torch.clamp(x, min=floor)

    def sqrt(self, x: Any) -> Any:
        return torch.sqrt(x)

    def broadcast_to(self, x: Any, shape: Any) -> Any:
        return torch.broadcast_to(x, tuple(shape))
