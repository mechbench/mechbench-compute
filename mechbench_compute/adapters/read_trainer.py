from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mechbench_compute.backends import backend_of


def read_trainer(model: Any) -> SimpleNamespace:
    if backend_of(model) == "torch":
        from mechbench_compute.torch_backend import lora_layers, training
        from mechbench_compute.torch_backend.determinism import read_training_numerics

        return SimpleNamespace(apply_lora=lora_layers.apply_lora, remove_lora=lora_layers.remove_lora,
                               read_adapter_bytes=lora_layers.read_adapter_bytes,
                               read_adapter_names=lambda m: list(lora_layers.read_lora_parameters(m)),
                               read_numerics=lambda m, det: read_training_numerics(m.device, det),
                               train_soft_ce=training.train_soft_ce)
    from mlx.utils import tree_flatten

    from mechbench_compute import lora
    from mechbench_compute.finetune import train_soft_ce

    return SimpleNamespace(
        apply_lora=lambda m, *a, **k: lora.apply_lora(m.lm, *a, **k),
        remove_lora=lambda m: lora.remove_lora(m.lm),
        read_adapter_bytes=lambda m: lora.read_adapter_bytes(m.lm),
        read_adapter_names=lambda m: [k for k, _ in tree_flatten(m.lm.trainable_parameters())],
        read_numerics=lambda m, det: None,
        train_soft_ce=lambda m, *a, deterministic=True, **k: train_soft_ce(m.lm, *a, **k))
