from __future__ import annotations

from typing import Any

import numpy as np

FRAMEWORKS: tuple[str, ...] = ("mlx", "torch", "numpy")


def read_framework(x: Any) -> str:
    if isinstance(x, (np.ndarray, np.generic)):
        return "numpy"
    root = type(x).__module__.split(".", 1)[0]
    if root in FRAMEWORKS:
        return root
    raise TypeError(
        f"compute reads MLX arrays, torch tensors and numpy arrays, not "
        f"{type(x).__module__}.{type(x).__qualname__}")


def read_f32(x: Any) -> np.ndarray:
    framework = read_framework(x)
    if framework == "mlx":
        from mechbench_compute._mlx import mx

        y = x.astype(mx.float32)
        mx.eval(y)
        return np.array(y)
    if framework == "torch":
        return x.detach().to(device="cpu").float().numpy()
    return np.asarray(x, dtype=np.float32)


def read_f64(x: Any) -> np.ndarray:
    framework = read_framework(x)
    if framework == "mlx":
        from mechbench_compute._mlx import mx

        y = x.astype(mx.float32)
        mx.eval(y)
        return np.array(y, dtype=np.float64)
    if framework == "torch":
        return read_f32(x).astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def make_f32(values: Any, like: Any) -> Any:
    framework = read_framework(like)
    if framework == "mlx":
        from mechbench_compute._mlx import mx

        return mx.array(values, dtype=mx.float32)
    if framework == "torch":
        import torch

        return torch.as_tensor(np.asarray(values, dtype=np.float32), device=like.device)
    return np.asarray(values, dtype=np.float32)
