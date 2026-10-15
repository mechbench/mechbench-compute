from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from functools import partial
from typing import Any

from mechbench_compute.torch_backend.loading import read_text_config
from mechbench_compute.torch_backend.memory import read_free_bytes

SWITCH = "MECHBENCH_GRADIENT_CHECKPOINTING"

SHARE_OF_FREE = 0.25


def estimate_activation_bytes(model: Any, tokens: int, *, math_attention: bool = False) -> int:
    cfg = read_text_config(model._model.config)
    d = int(cfg.hidden_size)
    width = getattr(cfg, "intermediate_size", None) or 4 * d
    ffn = int(max(width) if isinstance(width, (list, tuple)) else width)
    per_token_layer = 2 * (20 * d + 5 * ffn)
    if math_attention:
        per_token_layer += 4 * int(cfg.num_attention_heads) * int(tokens)
    logits = 6 * int(cfg.vocab_size)
    return int(tokens) * (int(cfg.num_hidden_layers) * per_token_layer + logits)


def decide_checkpointing(model: Any, longest: int, *, math_attention: bool = False) -> bool:
    chosen = os.environ.get(SWITCH, "auto").strip().lower()
    if chosen in ("on", "1", "true"):
        return True
    if chosen in ("off", "0", "false"):
        return False
    if chosen != "auto":
        raise ValueError(f"{SWITCH} is on, off or auto, not {chosen!r}")
    free = read_free_bytes(model.device)
    return free is not None and (estimate_activation_bytes(model, longest, math_attention=math_attention)
                                 > SHARE_OF_FREE * free)


def check_layers_checkpoint(layers: Any) -> None:
    from transformers.modeling_layers import GradientCheckpointingLayer

    plain = sorted({type(layer).__name__ for layer in layers
                    if not isinstance(layer, GradientCheckpointingLayer)})
    if plain:
        raise NotImplementedError(
            f"gradient checkpointing wraps each decoder layer, and {', '.join(plain)} "
            f"does not take it; set {SWITCH}=off")


# external: transformers modeling_layers.py — a GradientCheckpointingLayer checkpoints only while its own `training` flag is set; set it on the layer alone so its children (dropout) stay in eval
@contextlib.contextmanager
def checkpointed(model: Any, on: bool) -> Iterator[None]:
    if not on:
        yield
        return
    from torch.utils.checkpoint import checkpoint

    layers = list(model.lm.model.layers)
    check_layers_checkpoint(layers)
    for layer in layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        layer.training = True
    try:
        yield
    finally:
        for layer in layers:
            layer.gradient_checkpointing = False
            layer.training = False
            del layer._gradient_checkpointing_func
