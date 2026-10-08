from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mechbench_compute.torch_backend.operator_verbs import TorchVerbs


class TorchOps:
    framework = "torch"
    float32 = torch.float32

    def __init__(self, device: Any) -> None:
        self.device = device
        self.key = ("torch", str(device))
        self.verbs = TorchVerbs(device)

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

    def lift(self, y: Any) -> Any:
        if isinstance(y, torch.Tensor):
            return y
        return torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)

    def take(self, x: Any, indices: Any, axis: int) -> Any:
        return x.index_select(axis, indices.long())

    def put_along_axis(self, x: Any, indices: Any, values: Any, axis: int) -> Any:
        return x.scatter(axis, indices.long(), values.to(x.dtype))

    def count_nonfinite(self, where: Any, y: Any) -> int:
        return int((where & ~torch.isfinite(self.lift(y))).sum().item())

    def zeros(self, shape: Any) -> Any:
        return torch.zeros(tuple(shape), dtype=torch.float32, device=self.device)

    def logsumexp(self, x: Any) -> Any:
        return torch.logsumexp(x, dim=-1)

    def exp(self, x: Any) -> Any:
        return torch.exp(x)

    def grad(self, objective: Any, deltas: dict[str, Any]) -> dict[str, Any]:
        return self.value_and_grad(lambda ds: (objective(ds), None), deltas)[1]

    def value_and_grad(self, read_value: Any, deltas: dict[str, Any]) -> Any:
        from mechbench_compute.torch_backend.forward import tracking_gradients

        leaves = {n: d.detach().clone().requires_grad_(True) for n, d in deltas.items()}
        with tracking_gradients():
            value, aux = read_value(leaves)
            got = torch.autograd.grad(value, list(leaves.values()), allow_unused=True)
        grads = {n: torch.zeros_like(leaves[n]) if g is None else g
                 for (n, _), g in zip(leaves.items(), got, strict=True)}
        return (value.detach(), aux), grads

    def read_parameter(self, module: Any, attr: str) -> Any:
        return getattr(module, attr).data

    def write_parameter(self, module: Any, attr: str, value: Any) -> None:
        getattr(module, attr).data = value

    def settle(self, arrays: list[Any]) -> None:
        return None

    def read_moments(self, chunk: Any) -> tuple[float, float, float, float]:
        chunk = chunk.detach().float()
        return (float(chunk.sum()), float((chunk * chunk).sum()), float((chunk == 0).sum()),
                float(chunk.abs().max()))
