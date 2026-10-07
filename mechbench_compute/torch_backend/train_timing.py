from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mechbench_compute.distill import Example, soft_ce
from mechbench_compute.torch_backend.lora_layers import apply_lora, remove_lora
from mechbench_compute.torch_backend.training import train_measured


def make_timing_items(model: Any, items: int, prompt_tokens: int, outcomes: int,
                      rng: np.random.Generator) -> list[Example]:
    vocab = int(model.arch.vocab_size)
    out = []
    for _ in range(int(items)):
        ids = rng.integers(0, vocab, size=int(prompt_tokens)).tolist()
        chosen = rng.choice(vocab, size=min(int(outcomes), vocab), replace=False).tolist()
        out.append(Example(ids, [], {int(t): 1.0 / len(chosen) for t in chosen}))
    return out


def time_training(model: Any, *, steps: int = 5, warmup: int = 1, rank: int = 8, alpha: float = 16.0,
                  targets: Sequence[str] = ("q_proj", "v_proj"), items: int = 6,
                  prompt_tokens: int = 128, outcomes: int = 6, lr: float = 1e-4, seed: int = 0,
                  checkpointing: bool | None = None,
                  project: Sequence[int] = (60, 240, 720)) -> dict[str, Any]:
    examples = make_timing_items(model, items, prompt_tokens, outcomes, np.random.default_rng(seed))
    groups, batch = {"target": examples}, {"target": len(examples)}
    n = apply_lora(model, int(rank), float(alpha), targets=tuple(targets), seed=seed,
                   keys=model.architecture.adapter_keys)
    try:
        if warmup:
            train_measured(model, groups, batch, steps=int(warmup), lr=lr, seed=seed,
                           loss_fn=soft_ce, checkpointing=checkpointing)
        loss, measured = train_measured(model, groups, batch, steps=int(steps), lr=lr, seed=seed,
                                        loss_fn=soft_ce, checkpointing=checkpointing)
    finally:
        remove_lora(model)
    per_step = measured["seconds_per_step"]
    peak = measured["peak_memory_bytes"]
    return {
        **measured, "model_type": model.arch.model_type, "rank": int(rank), "targets": list(targets),
        "lora_params": n, "items": int(items), "prompt_tokens": int(prompt_tokens),
        "warmup_steps": int(warmup), "final_loss": round(loss, 6),
        "peak_memory_gb": round(peak / 2**30, 3) if peak is not None else None,
        "projected_seconds": {int(k): round(per_step * int(k), 1) for k in project} if per_step else {},
    }
