from __future__ import annotations

import os
from typing import Any

CUBLAS_WORKSPACE = ":4096:8"


def make_deterministic(device: Any) -> None:
    import torch

    if getattr(device, "type", str(device)) != "cuda":
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
    if getattr(device, "type", None) == "cuda":
        out.update(cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                   cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    return out
