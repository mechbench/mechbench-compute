from __future__ import annotations

from typing import Any

from mechbench_compute._mlx import mx
from mechbench_compute.arrays import read_framework


class MlxOps:
    framework = "mlx"
    key = "mlx"
    verbs = mx

    @property
    def float32(self) -> Any:
        return mx.float32

    def array(self, values: Any) -> Any:
        return mx.array(values)

    def cast(self, x: Any, dtype: Any) -> Any:
        return x.astype(dtype)

    def size(self, x: Any) -> int:
        return x.size

    def sum(self, x: Any, axis: int) -> Any:
        return mx.sum(x, axis=axis, keepdims=True)

    def full(self, shape: Any, value: float, dtype: Any) -> Any:
        return mx.full(shape, value, dtype)

    def zeros_like(self, x: Any) -> Any:
        return mx.zeros_like(x)

    def clip(self, x: Any, lo: float, hi: float) -> Any:
        return mx.clip(x, lo, hi)

    def where(self, mask: Any, a: Any, b: Any) -> Any:
        return mx.where(mask, a, b)

    def maximum(self, x: Any, floor: float) -> Any:
        return mx.maximum(x, floor)

    def sqrt(self, x: Any) -> Any:
        return mx.sqrt(x)

    def broadcast_to(self, x: Any, shape: Any) -> Any:
        return mx.broadcast_to(x, shape)

    def lift(self, y: Any) -> Any:
        return y if isinstance(y, mx.array) else mx.array(y, dtype=mx.float32)

    def take(self, x: Any, indices: Any, axis: int) -> Any:
        return mx.take(x, indices, axis=axis)

    def put_along_axis(self, x: Any, indices: Any, values: Any, axis: int) -> Any:
        return mx.put_along_axis(x, indices, values, axis=axis)

    def count_nonfinite(self, where: Any, y: Any) -> int:
        return int(mx.sum(mx.logical_and(where, mx.logical_not(mx.isfinite(y)))).item())

    def zeros(self, shape: Any) -> Any:
        return mx.zeros(shape, dtype=mx.float32)

    def logsumexp(self, x: Any) -> Any:
        return mx.logsumexp(x)

    def exp(self, x: Any) -> Any:
        return mx.exp(x)

    def grad(self, objective: Any, deltas: dict[str, Any]) -> dict[str, Any]:
        grads = mx.grad(objective)(deltas)
        mx.eval(*grads.values())
        return grads

    def value_and_grad(self, read_value: Any, deltas: dict[str, Any]) -> Any:
        return mx.value_and_grad(read_value)(deltas)


MLX_OPS = MlxOps()


def read_array_ops(act: Any) -> Any:
    if read_framework(act) == "torch":
        from mechbench_compute.torch_backend.array_ops import TorchOps

        return TorchOps(act.device)
    return MLX_OPS
