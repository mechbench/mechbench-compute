from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from mechbench_compute import lexicon
from mechbench_compute.adapters.read_trainer import read_trainer
from mechbench_compute.distill import soft_ce
from mechbench_compute.lexicon._base import DEFAULT_OUTPUT


def train_lora(ctx, model, params, groups: Mapping[str, list[Any]], batch: Mapping[str, int],
               methods: dict[str, Any], *, trained_on: dict[str, Any],
               factories: Mapping[str, Callable] | None = None,
               loss_fn: Callable = soft_ce) -> dict[str, Any]:
    lora_cfg = params.get("lora") or {}
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16))
    target_modules = tuple(lora_cfg.get("target_modules")
                           or ("q_proj", "v_proj"))
    steps, lr, seed = int(methods["steps"]), float(methods["lr"]), int(methods["seed"])
    checkpoint_every = int(params.get("checkpoint_every", 50))
    kept_steps = read_kept_steps(params, steps, checkpoint_every)
    trainer = read_trainer(model)

    def keep(step, loss, data):
        return {"id": f"step-{step}", "coords": {"step": step}, **lineage,
                "train": methods, "loss": float(loss), "data": data}

    def on_step(step, loss):
        if step in kept_steps and step < steps:
            kept[step] = keep(step, loss, trainer.read_adapter_bytes(model))
            if ctx.on_item:
                ctx.on_item(kept[step]["id"], kept[step], False)
        elif ctx.on_item:
            ctx.on_item()

    try:
        n_lora = trainer.apply_lora(model, rank, alpha, targets=target_modules,
                                    seed=seed, keys=model.architecture.adapter_keys)
        lora = {"rank": rank, "alpha": alpha, "scale": alpha / rank,
                "target_modules": list(target_modules), "params": n_lora}
        lineage = {"kind": "adapter/lora", "format": "safetensors",
                   "base_model": trained_on["base"], "trained_on": trained_on,
                   "lora": lora}
        if kept_steps:
            check_kept_size(len(trainer.read_adapter_bytes(model)), len(kept_steps))
        if ctx.on_start:
            ctx.on_start(steps)
        resumed_from = int(ctx.resume_state["step"]) if ctx.resume_state else 0
        kept = restore_kept(kept_steps, resumed_from, ctx.resume_items)
        if resumed_from and ctx.on_item:
            for step in range(1, resumed_from + 1):
                if step in kept:
                    ctx.on_item(kept[step]["id"], kept[step], True)
                else:
                    ctx.on_item(None, None, True)
        final_loss = trainer.train_soft_ce(
            model, groups, batch, steps=steps, lr=lr, seed=seed,
            factories=factories,
            on_step=on_step if (ctx.on_item or kept_steps) else None,
            checkpoint_every=checkpoint_every if ctx.on_checkpoint else 0,
            on_checkpoint=ctx.on_checkpoint,
            resume_state=ctx.resume_state,
            loss_fn=loss_fn)
        data = trainer.read_adapter_bytes(model)
    finally:
        try:
            trainer.remove_lora(model)
        finally:
            ctx.evict_model()

    adapter = {**lineage,
               "train": {**methods, "final_loss": round(final_loss, 4)},
               "data": data}
    if not kept_steps:
        return {DEFAULT_OUTPUT: adapter}
    kept[steps] = keep(steps, final_loss, data)
    return {DEFAULT_OUTPUT: adapter,
            "checkpoints": lexicon.collection(
                "adapter/lora", [kept[s] for s in kept_steps],
                base_model=adapter["base_model"], trained_on=trained_on,
                lora=lora, train=adapter["train"],
                checkpoint_every=checkpoint_every)}


def read_kept_steps(params, steps: int, every: int) -> list[int]:
    if not params.get("keep_checkpoints", False):
        return []
    if every <= 0:
        raise ValueError(
            "adapter/train: keep_checkpoints keeps an adapter every "
            "`checkpoint_every` steps; set it above 0")
    return [*range(every, steps, every), steps]


def check_kept_size(adapter_bytes: int, count: int) -> None:
    from mechbench_compute.bench import MAX_OBJECT_BYTES

    if adapter_bytes * count > MAX_OBJECT_BYTES:
        raise ValueError(
            f"adapter/train: {count} kept checkpoints of {adapter_bytes:,} bytes "
            f"each are over the {MAX_OBJECT_BYTES:,}-byte object limit the "
            f"collection is stored under; keep fewer with a larger "
            f"`checkpoint_every` (at most {max(1, MAX_OBJECT_BYTES // adapter_bytes)} fit)")


def restore_kept(kept_steps: list[int], resumed_from: int,
                 resume_items) -> dict[int, dict]:
    kept: dict[int, dict] = {}
    for step in (s for s in kept_steps if s <= resumed_from):
        item = (resume_items or {}).get(f"step-{step}")
        if item is None:
            raise ValueError(
                f"adapter/train: resuming at step {resumed_from}, but the "
                f"adapter kept at step {step} is not among the run's kept "
                f"items; restart the node")
        kept[step] = item
    return kept
