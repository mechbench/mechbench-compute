from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

CUBLAS_WORKSPACE = ":4096:8"

# external: cuBLAS — only these two workspace configurations make its reductions repeat bit for bit
DETERMINISTIC_WORKSPACES = (":4096:8", ":16:8")

NOT_DETERMINISTIC = (
    "trained with `deterministic: false`: torch was not held to deterministic algorithms, so on "
    "CUDA attention's backward and the scatter and index-add backwards may add in a different "
    "order each run, and two runs with the same seed can differ in their last bits")


def is_cuda(device: Any) -> bool:
    return getattr(device, "type", str(device)) == "cuda"


def make_deterministic(device: Any) -> None:
    import torch

    if not is_cuda(device):
        return
    # external: cuBLAS — CUBLAS_WORKSPACE_CONFIG must be set before the first cuBLAS handle for its reductions to repeat bit for bit
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def read_numerics(device: Any) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "threads": torch.get_num_threads(),
    }
    if is_cuda(device):
        out.update(cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                   cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    return out


def read_training_numerics(device: Any, deterministic: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"deterministic": bool(deterministic), **read_numerics(device)}
    if deterministic:
        out.update(deterministic_algorithms=True, deterministic_warn_only=False)
    if is_cuda(device):
        out["attention_backward"] = "math" if deterministic else "default"
    if not deterministic:
        out["note"] = NOT_DETERMINISTIC
    return out


def check_workspace(device: Any) -> None:
    if not is_cuda(device):
        return
    held = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if held not in DETERMINISTIC_WORKSPACES:
        raise ValueError(
            f"adapter/train: deterministic training needs CUBLAS_WORKSPACE_CONFIG "
            f"{' or '.join(DETERMINISTIC_WORKSPACES)} set before CUDA starts, and it is {held!r}; "
            f"unset it, or train with `deterministic: false`")


@contextlib.contextmanager
def training_numerics(device: Any, deterministic: bool) -> Iterator[None]:
    import torch

    if not deterministic:
        yield
        return
    check_workspace(device)
    held = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        if is_cuda(device):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                yield
        else:
            yield
    finally:
        torch.use_deterministic_algorithms(held[0], warn_only=held[1])
