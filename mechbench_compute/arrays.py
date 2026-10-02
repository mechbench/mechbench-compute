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


def zeros_like(x: Any) -> Any:
    framework = read_framework(x)
    if framework == "mlx":
        from mechbench_compute._mlx import mx

        return mx.zeros_like(x)
    if framework == "torch":
        import torch

        return torch.zeros_like(x)
    return np.zeros_like(x)


def make_head_mask(act: Any, head: int) -> Any:
    n_heads = act.shape[1]
    if read_framework(act) == "mlx":
        from mechbench_compute._mlx import mx

        mask = mx.ones((1, n_heads, 1, 1))
        return mask.at[:, head, :, :].add(-1.0)
    import torch

    mask = torch.ones((1, n_heads, 1, 1), dtype=torch.float32, device=act.device)
    mask[:, head, :, :] = 0.0
    return mask


def make_position_mask(act: Any, position: int) -> Any:
    seq_len = act.shape[1]
    if read_framework(act) == "mlx":
        from mechbench_compute._mlx import mx

        mask = mx.zeros((1, seq_len, 1), dtype=act.dtype)
        return mask.at[:, position, :].add(1.0)
    import torch

    mask = torch.zeros((1, seq_len, 1), dtype=act.dtype, device=act.device)
    mask[:, position, :] = 1.0
    return mask


def match_framework(value: Any, like: Any) -> Any:
    if read_framework(like) != "torch":
        return value
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=like.device)
    if read_framework(value) == "mlx":
        value = read_f32(value)
    return torch.as_tensor(np.asarray(value), device=like.device)


def read_logprobs(x: Any) -> np.ndarray:
    framework = read_framework(x)
    if framework == "mlx":
        from mechbench_compute._mlx import mx

        last = x.astype(mx.float32)
        lp = last - mx.logsumexp(last)
        mx.eval(lp)
        return np.array(lp)
    if framework == "torch":
        import torch

        last = x.detach().float()
        return (last - torch.logsumexp(last, dim=-1, keepdim=True)).cpu().numpy()
    last = np.asarray(x, dtype=np.float32)
    top = last.max(axis=-1, keepdims=True)
    return last - (top + np.log(np.exp(last - top).sum(axis=-1, keepdims=True)))


def read_softmax(x: Any) -> np.ndarray:
    framework = read_framework(x)
    if framework == "mlx":
        from mechbench_compute._mlx import mx

        probs = mx.softmax(x.astype(mx.float32))
        mx.eval(probs)
        return np.array(probs)
    if framework == "torch":
        import torch

        return torch.softmax(x.detach().float(), dim=-1).cpu().numpy()
    return np.exp(read_logprobs(x))
