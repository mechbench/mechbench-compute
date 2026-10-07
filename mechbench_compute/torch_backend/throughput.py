from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from mechbench_compute.torch_backend.batched import forward_rows, left_pad
from mechbench_compute.torch_backend.decoding import make_prompt_cache


def synchronize(device: Any) -> None:
    import torch

    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def measure_once(model: Any, batch: int, prompt_tokens: int, new_tokens: int,
                 rng: np.random.Generator) -> dict[str, Any]:
    import torch

    vocab = int(model.arch.vocab_size)
    prompts = [rng.integers(0, vocab, size=int(prompt_tokens)).tolist() for _ in range(batch)]
    device = model.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ids, mask, positions = left_pad(prompts, 0, device)
    cache = make_prompt_cache(model._model)
    synchronize(device)
    started = time.perf_counter()
    rows = forward_rows(model, ids, mask, positions, cache)
    synchronize(device)
    prefilled = time.perf_counter()
    last = positions[:, -1:]
    for _ in range(int(new_tokens)):
        step = rows.argmax(dim=-1, keepdim=True)
        last = last + 1
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        rows = forward_rows(model, step, mask, last, cache)
    synchronize(device)
    decoded = time.perf_counter()
    prefill_s, decode_s = prefilled - started, decoded - prefilled
    return {
        "batch": batch, "prompt_tokens": int(prompt_tokens), "new_tokens": int(new_tokens),
        "prefill_seconds": round(prefill_s, 6), "decode_seconds": round(decode_s, 6),
        "prefill_tokens_per_s": round(batch * prompt_tokens / prefill_s, 2) if prefill_s else None,
        "decode_tokens_per_s": round(batch * new_tokens / decode_s, 2) if decode_s else None,
        "peak_memory_gb": (round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                           if device.type == "cuda" else None),
    }


def measure_throughput(model: Any, *, batch_sizes: Sequence[int] = (1, 4, 16),
                       prompt_tokens: int = 128, new_tokens: int = 64, warmup: bool = True,
                       seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if warmup:
        measure_once(model, 1, min(8, int(prompt_tokens)), 2, rng)
    return [{**measure_once(model, int(b), prompt_tokens, new_tokens, rng),
             "accelerator": model.accelerator, "model_type": model.arch.model_type}
            for b in batch_sizes]
