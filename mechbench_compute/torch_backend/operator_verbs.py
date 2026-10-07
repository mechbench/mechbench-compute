from __future__ import annotations

from typing import Any

import torch


class TorchVerbs:
    def __init__(self, device: Any) -> None:
        self.device = device

    def array(self, value: Any) -> Any:
        return self.lift(value)

    def lift(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, bool):
            return torch.tensor(value, device=self.device)
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

    def ones_like(self, x: Any) -> Any:
        return torch.ones_like(self.lift(x))

    def add(self, a: Any, b: Any) -> Any:
        return torch.add(self.lift(a), self.lift(b))

    def subtract(self, a: Any, b: Any) -> Any:
        return torch.subtract(self.lift(a), self.lift(b))

    def multiply(self, a: Any, b: Any) -> Any:
        return torch.multiply(self.lift(a), self.lift(b))

    def divide(self, a: Any, b: Any) -> Any:
        return torch.divide(self.lift(a), self.lift(b))

    def floor_divide(self, a: Any, b: Any) -> Any:
        return torch.floor(torch.divide(self.lift(a), self.lift(b)))

    def remainder(self, a: Any, b: Any) -> Any:
        return torch.remainder(self.lift(a), self.lift(b))

    def power(self, a: Any, b: Any) -> Any:
        return torch.pow(self.lift(a), self.lift(b))

    def equal(self, a: Any, b: Any) -> Any:
        return torch.eq(self.lift(a), self.lift(b))

    def not_equal(self, a: Any, b: Any) -> Any:
        return torch.ne(self.lift(a), self.lift(b))

    def less(self, a: Any, b: Any) -> Any:
        return torch.lt(self.lift(a), self.lift(b))

    def less_equal(self, a: Any, b: Any) -> Any:
        return torch.le(self.lift(a), self.lift(b))

    def greater(self, a: Any, b: Any) -> Any:
        return torch.gt(self.lift(a), self.lift(b))

    def greater_equal(self, a: Any, b: Any) -> Any:
        return torch.ge(self.lift(a), self.lift(b))

    def logical_and(self, a: Any, b: Any) -> Any:
        return torch.logical_and(self.lift(a), self.lift(b))

    def logical_or(self, a: Any, b: Any) -> Any:
        return torch.logical_or(self.lift(a), self.lift(b))

    def logical_not(self, a: Any) -> Any:
        return torch.logical_not(self.lift(a))

    def negative(self, a: Any) -> Any:
        return torch.negative(self.lift(a))

    def abs(self, a: Any) -> Any:
        return torch.abs(self.lift(a))

    def ceil(self, a: Any) -> Any:
        return torch.ceil(self.lift(a))

    def floor(self, a: Any) -> Any:
        return torch.floor(self.lift(a))

    def round(self, a: Any) -> Any:
        return torch.round(self.lift(a))

    def exp(self, a: Any) -> Any:
        return torch.exp(self.lift(a))

    def log(self, a: Any) -> Any:
        return torch.log(self.lift(a))

    def log2(self, a: Any) -> Any:
        return torch.log2(self.lift(a))

    def log10(self, a: Any) -> Any:
        return torch.log10(self.lift(a))

    def sqrt(self, a: Any) -> Any:
        return torch.sqrt(self.lift(a))

    def minimum(self, a: Any, b: Any) -> Any:
        return torch.minimum(self.lift(a), self.lift(b))

    def maximum(self, a: Any, b: Any) -> Any:
        return torch.maximum(self.lift(a), self.lift(b))

    def where(self, c: Any, a: Any, b: Any) -> Any:
        return torch.where(self.lift(c), self.lift(a), self.lift(b))
