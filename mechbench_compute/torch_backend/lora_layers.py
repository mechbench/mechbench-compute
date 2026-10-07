from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from mechbench_compute.adapter_keys import ADAPTER_KEYS, AdapterKeys
from mechbench_compute.torch_backend.mlx_random import draw_normal, make_key, split_key


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, r: int, alpha: float, key: np.ndarray | None = None):
        super().__init__()
        self.base = base
        out_dim, in_dim = base.weight.shape
        self.scale = alpha / r
        draw = (torch.from_numpy(draw_normal((r, in_dim), key)) if key is not None
                else torch.randn((r, in_dim), dtype=torch.float32))
        device = base.weight.device
        self.lora_a = torch.nn.Parameter((draw * np.float32(1.0 / math.sqrt(in_dim))).to(device))
        self.lora_b = torch.nn.Parameter(torch.zeros((out_dim, r), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        z = (x.float() @ self.lora_a.T) @ self.lora_b.T
        return y + (self.scale * z).to(y.dtype)


def apply_lora(model: Any, rank: int = 8, alpha: float = 16.0,
               targets: tuple[str, ...] = ("q_proj", "v_proj"), *, seed: int | None = None,
               keys: AdapterKeys = ADAPTER_KEYS) -> int:
    lm = model.lm
    model.frozen = {name: p.requires_grad for name, p in model._model.named_parameters()}
    for p in model._model.parameters():
        p.requires_grad_(False)
    n = 0
    key = make_key(seed) if seed is not None else None
    wrapped_per_target = dict.fromkeys(targets, 0)
    try:
        for layer in lm.model.layers:
            for name in targets:
                container = keys.containers.get(name)
                if container is None:
                    raise ValueError(f"unknown target module {name!r}; known: {sorted(keys.containers)}")
                holder = getattr(layer, container)
                base = getattr(holder, name, None)
                if base is None:
                    continue
                sub = None
                if key is not None:
                    key, sub = split_key(key)
                wrapped = LoRALinear(base, rank, alpha, key=sub)
                setattr(holder, name, wrapped)
                wrapped_per_target[name] += 1
                n += wrapped.lora_a.numel() + wrapped.lora_b.numel()
        dead = [t for t, c in wrapped_per_target.items() if c == 0]
        if dead:
            raise ValueError(
                f"target modules {dead!r} exist on no layer of this architecture — adapter would "
                f"train nothing for them. Wrapped counts: {wrapped_per_target!r}")
    except BaseException:
        remove_lora(model)
        raise
    return n


def remove_lora(model: Any) -> int:
    removed = 0
    for module in list(model.lm.model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.base)
                removed += 1
    frozen = getattr(model, "frozen", None) or {}
    for name, p in model._model.named_parameters():
        p.requires_grad_(frozen.get(name, p.requires_grad))
    model.frozen = None
    return removed


def read_lora_parameters(model: Any) -> dict[str, torch.nn.Parameter]:
    keys = model.architecture.adapter_keys
    out: dict[str, torch.nn.Parameter] = {}
    for i, layer in enumerate(model.lm.model.layers):
        for proj, container in keys.containers.items():
            found = getattr(getattr(layer, container, None), proj, None)
            if isinstance(found, LoRALinear):
                stem = f"model.layers.{i}.{container}.{proj}"
                out[f"{stem}.lora_a"] = found.lora_a
                out[f"{stem}.lora_b"] = found.lora_b
    return dict(sorted(out.items()))


def read_lora_arrays(model: Any) -> dict[str, np.ndarray]:
    return {k: p.detach().cpu().numpy().copy() for k, p in read_lora_parameters(model).items()}


def read_adapter_bytes(model: Any) -> bytes:
    from safetensors.torch import save

    return save({k: p.detach().cpu().contiguous() for k, p in read_lora_parameters(model).items()})
