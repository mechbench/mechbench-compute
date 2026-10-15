from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from mechbench_compute import lexicon
from mechbench_compute.adapters.read_lora_layers import read_lora_layers
from mechbench_compute.adapters.read_trainer import read_trainer
from mechbench_compute.distill import soft_ce
from mechbench_compute.lexicon._base import DEFAULT_OUTPUT

LAYER_OF = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def train_lora(ctx, model, params, groups: Mapping[str, list[Any]], batch: Mapping[str, int],
               methods: dict[str, Any], *, trained_on: dict[str, Any],
               factories: Mapping[str, Callable] | None = None,
               loss_fn: Callable = soft_ce) -> dict[str, Any]:
    lora_cfg = params.get("lora") or {}
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16))
    target_modules = tuple(lora_cfg.get("target_modules")
                           or ("q_proj", "v_proj"))
    n_layers = len(model.lm.model.layers)
    layers = read_lora_layers(lora_cfg.get("layers", "all"), n_layers)
    every = len(layers) == n_layers
    steps, lr, seed = int(methods["steps"]), float(methods["lr"]), int(methods["seed"])
    checkpoint_every = int(params.get("checkpoint_every", 50))
    kept_steps = read_kept_steps(params, steps, checkpoint_every)
    trainer = read_trainer(model)
    deterministic = bool(params.get("deterministic", True))
    numerics = trainer.read_numerics(model, deterministic)
    if numerics is not None:
        methods = {**methods, "numerics": numerics}

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
                                    seed=seed, keys=model.architecture.adapter_keys,
                                    layers=None if every else layers)
        lora = {"rank": rank, "alpha": alpha, "scale": alpha / rank,
                "target_modules": list(target_modules),
                **({} if every else {"layers": layers}), "params": n_lora}
        if ctx.resume_state and "weights" in ctx.resume_state:
            check_resume_state(ctx.resume_state, trainer.read_adapter_names(model))
        lineage = {"kind": "adapter/lora", "format": "safetensors",
                   "base_model": trained_on["base"], "trained_on": trained_on,
                   "lora": lora}
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
            loss_fn=loss_fn, deterministic=deterministic)
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


def check_resume_state(state, names: list[str]) -> None:
    saved = sorted(state["weights"])
    if saved != sorted(names):
        raise ValueError(
            f"adapter/train: the saved training state holds {describe_weights(saved)} and this "
            f"node's adapter {describe_weights(names)}: its `lora` differs from the run that "
            f"saved it; restart the node")


def describe_weights(names: list[str]) -> str:
    layers = sorted({int(m.group(1)) for n in names if (m := LAYER_OF.search(n))})
    return f"{len(names)} weights on layers {', '.join(map(str, layers))}"


def read_kept_steps(params, steps: int, every: int) -> list[int]:
    if not params.get("keep_checkpoints", False):
        return []
    if every <= 0:
        raise ValueError(
            "adapter/train: keep_checkpoints keeps an adapter every "
            "`checkpoint_every` steps; set it above 0")
    return [*range(every, steps, every), steps]


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
