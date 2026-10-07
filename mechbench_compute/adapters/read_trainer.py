from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mechbench_compute.backends import backend_of


def read_trainer(model: Any) -> SimpleNamespace:
    if backend_of(model) == "torch":
        from mechbench_compute.torch_backend import lora_layers, training

        return SimpleNamespace(apply_lora=lora_layers.apply_lora, remove_lora=lora_layers.remove_lora,
                               read_adapter_bytes=lora_layers.read_adapter_bytes,
                               train_soft_ce=training.train_soft_ce)
    from mechbench_compute import lora
    from mechbench_compute.finetune import train_soft_ce

    return SimpleNamespace(
        apply_lora=lambda m, *a, **k: lora.apply_lora(m.lm, *a, **k),
        remove_lora=lambda m: lora.remove_lora(m.lm),
        read_adapter_bytes=lambda m: lora.read_adapter_bytes(m.lm),
        train_soft_ce=lambda m, *a, **k: train_soft_ce(m.lm, *a, **k))
