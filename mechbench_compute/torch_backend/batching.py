from __future__ import annotations

from typing import Any

from mechbench_compute.torch_backend.loading import read_text_config

MEMORY_SHARE = 0.5


def read_row_bytes(model: Any, tokens: int, *, logits_rows: int = 1) -> int:
    cfg = read_text_config(model._model.config)
    heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or int(cfg.hidden_size) // heads)
    kv_heads = int(getattr(cfg, "num_key_value_heads", None) or heads)
    width = next(model._model.parameters()).element_size()
    kv = 2 * int(cfg.num_hidden_layers) * kv_heads * head_dim * width * int(tokens)
    hidden = 4 * int(cfg.hidden_size) * width * int(tokens)
    logits = int(cfg.vocab_size) * 4 * 2 * int(logits_rows)
    return kv + hidden + logits


def read_free_bytes(device: Any) -> int | None:
    if getattr(device, "type", None) != "cuda":
        return None
    import torch

    free, _ = torch.cuda.mem_get_info(device)
    return int(free)


def bound_batch(model: Any, requested: int, tokens: int, *, logits_rows: int = 1) -> int:
    requested = max(1, int(requested))
    free = read_free_bytes(model.device)
    if free is None:
        return requested
    per_row = max(1, read_row_bytes(model, tokens, logits_rows=logits_rows))
    return max(1, min(requested, int(free * MEMORY_SHARE) // per_row))


def chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), max(1, int(size)))]
