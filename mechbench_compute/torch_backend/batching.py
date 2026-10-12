from __future__ import annotations

from typing import Any

from mechbench_compute.torch_backend.loading import read_text_config
from mechbench_compute.torch_backend.memory import read_free_bytes

MEMORY_SHARE = 0.5


# external: transformers 5.19 — a config attribute that differs between layers raises when read whole (Gemma 4's head_dim, 31B's num_key_value_heads); per_layer_config carries each layer's own
def read_kv_widths(cfg: Any) -> list[int]:
    heads = int(cfg.num_attention_heads)
    layers = getattr(cfg, "per_layer_config", None) or [cfg] * int(cfg.num_hidden_layers)
    return [int(getattr(layer, "num_key_value_heads", None) or heads)
            * int(getattr(layer, "head_dim", None) or int(cfg.hidden_size) // heads)
            for layer in layers]


def read_row_bytes(model: Any, tokens: int, *, logits_rows: int = 1) -> int:
    cfg = read_text_config(model._model.config)
    width = next(model._model.parameters()).element_size()
    kv = 2 * sum(read_kv_widths(cfg)) * width * int(tokens)
    hidden = 4 * int(cfg.hidden_size) * width * int(tokens)
    logits = int(cfg.vocab_size) * 4 * 2 * int(logits_rows)
    return kv + hidden + logits



def bound_batch(model: Any, requested: int, tokens: int, *, logits_rows: int = 1) -> int:
    requested = max(1, int(requested))
    free = read_free_bytes(model.device)
    if free is None:
        return requested
    per_row = max(1, read_row_bytes(model, tokens, logits_rows=logits_rows))
    return max(1, min(requested, int(free * MEMORY_SHARE) // per_row))


def chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), max(1, int(size)))]
