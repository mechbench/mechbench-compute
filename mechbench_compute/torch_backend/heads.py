from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mechbench_compute.support import Unembed
from mechbench_compute.torch_backend.norms import TorchNorm


def make_projection(head: Any) -> Callable[[Any], Any]:
    def project(x: Any) -> Any:
        import torch

        return torch.nn.functional.linear(x, head.weight.to(x.dtype), head.bias)

    return project


def cap_logits(unembed: Unembed, logits: Any) -> Any:
    if unembed.softcap is None:
        return logits
    import torch

    return torch.tanh(logits / unembed.softcap) * unembed.softcap


def make_unembed(read_text: Callable[[Any], Any], gain_offset: float) -> Callable[[Any], Unembed]:
    def read_unembed(model: Any) -> Unembed:
        cfg = getattr(model.config, "text_config", None) or model.config
        return Unembed(norm=TorchNorm(read_text(model).norm, gain_offset),
                       project=make_projection(model.lm_head),
                       softcap=getattr(cfg, "final_logit_softcapping", None))

    return read_unembed


def make_head_logits(read_unembed: Callable[[Any], Unembed]) -> Callable[[Any, Any], Any]:
    def head_logits(model: Any, hidden: Any) -> Any:
        u = read_unembed(model)
        return cap_logits(u, u.project(hidden))

    return head_logits


def make_project_to_logits(read_unembed: Callable[[Any], Unembed]) -> Callable[[Any, Any], Any]:
    def project_to_logits(model: Any, residual: Any) -> Any:
        u = read_unembed(model)
        return cap_logits(u, u.project(u.norm(residual)))

    return project_to_logits

