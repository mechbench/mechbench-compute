from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from mechbench_compute.adapters.sample_batch import read_draws, sample_batch
from mechbench_compute.spans import add_to_span, record_training
from mechbench_compute.torch_backend.checkpointing import (
    checkpointed,
    decide_checkpointing,
)
from mechbench_compute.torch_backend.lora_layers import read_lora_parameters
from mechbench_compute.torch_backend.throughput import synchronize
from mechbench_compute.torch_backend.training_losses import TorchLoss, read_torch_loss


class Adam:
    def __init__(self, params: Mapping[str, torch.nn.Parameter], lr: float,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.params = dict(params)
        self.lr = float(lr)
        self.betas = betas
        self.eps = eps
        self.step = 0
        self.m = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}

    # external: mlx.optimizers.Adam — bias_correction is off by default, so the update is lr·m / (√v + ε)
    @torch.no_grad()
    def update(self) -> None:
        b1, b2 = self.betas
        lr = torch.tensor(self.lr, dtype=torch.float32)
        for k, p in self.params.items():
            g = p.grad
            if g is None:
                g = torch.zeros_like(p)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * torch.square(g)
            p.copy_(p - lr.to(p.device) * self.m[k] / (torch.sqrt(self.v[k]) + self.eps))
            p.grad = None
        self.step += 1

    def read_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {"step": np.array(self.step, dtype=np.uint64),
                               "learning_rate": np.array(self.lr, dtype=np.float32)}
        for k in self.params:
            out[f"{k}.m"] = self.m[k].detach().cpu().numpy().copy()
            out[f"{k}.v"] = self.v[k].detach().cpu().numpy().copy()
        return out

    def write_state(self, state: Mapping[str, Any]) -> None:
        self.step = int(np.asarray(state["step"]))
        for k, p in self.params.items():
            self.m[k] = torch.as_tensor(np.asarray(state[f"{k}.m"]), dtype=p.dtype).to(p.device)
            self.v[k] = torch.as_tensor(np.asarray(state[f"{k}.v"]), dtype=p.dtype).to(p.device)


def capture_training_state(params: Mapping[str, torch.nn.Parameter], opt: Adam, step: int,
                           rng: np.random.Generator) -> dict[str, Any]:
    return {"step": int(step),
            "weights": {k: p.detach().cpu().numpy().copy() for k, p in params.items()},
            "opt_state": opt.read_state(),
            "np_rng": rng.bit_generator.state,
            "mx_key": None}


def restore_training_state(params: Mapping[str, torch.nn.Parameter], opt: Adam,
                           rng: np.random.Generator, state: Mapping[str, Any]) -> int:
    missing = sorted(set(params) - set(state["weights"]))
    if missing:
        raise ValueError(f"adapter/train: the saved training state lacks {missing[:3]}; restart the node")
    with torch.no_grad():
        for k, p in params.items():
            p.copy_(torch.as_tensor(np.asarray(state["weights"][k]), dtype=p.dtype).to(p.device))
    opt.write_state(state["opt_state"])
    rng.bit_generator.state = state["np_rng"]
    return int(state["step"])


def step_once(model: Any, batch: list[Any], loss: TorchLoss) -> float:
    count = sum(loss.count(ex) for ex in batch)
    if count == 0:
        raise ValueError("adapter/train: a step drew nothing to train")
    total = torch.zeros((), dtype=torch.float32, device=model.device)
    for ex in batch:
        part = loss.total(model, ex)
        (part / count).backward()
        total = total + part.detach()
    return float(total / count)


def train_soft_ce(model: Any, groups: Mapping[str, list[Any]], batch_sizes: Mapping[str, int], *,
                  steps: int, lr: float = 1e-4, seed: int = 7,
                  factories: Mapping[str, Callable[[np.random.Generator], list[Any]]] | None = None,
                  on_step: Callable[[int, float], None] | None = None, checkpoint_every: int = 0,
                  on_checkpoint: Callable[[dict], None] | None = None,
                  resume_state: Mapping | None = None, loss_fn: Callable | None = None,
                  checkpointing: bool | None = None) -> float:
    loss_val, measured = train_measured(
        model, groups, batch_sizes, steps=steps, lr=lr, seed=seed, factories=factories,
        on_step=on_step, checkpoint_every=checkpoint_every, on_checkpoint=on_checkpoint,
        resume_state=resume_state, loss_fn=loss_fn, checkpointing=checkpointing)
    record_training(**measured)
    return loss_val


def train_measured(model: Any, groups: Mapping[str, list[Any]], batch_sizes: Mapping[str, int], *,
                   steps: int, lr: float = 1e-4, seed: int = 7,
                   factories: Mapping[str, Callable[[np.random.Generator], list[Any]]] | None = None,
                   on_step: Callable[[int, float], None] | None = None, checkpoint_every: int = 0,
                   on_checkpoint: Callable[[dict], None] | None = None,
                   resume_state: Mapping | None = None, loss_fn: Callable | None = None,
                   checkpointing: bool | None = None) -> tuple[float, dict[str, Any]]:
    loss = read_torch_loss(loss_fn)
    rng = np.random.default_rng(seed)
    params = read_lora_parameters(model)
    opt = Adam(params, lr)
    start = 1
    if resume_state is not None:
        start = restore_training_state(params, opt, rng, resume_state) + 1
    active, active_factories = read_draws(groups, batch_sizes, factories)
    drawn = [ex for _, f, _ in active_factories for ex in f(np.random.default_rng(seed))]
    longest = max((loss.tokens(ex) for ex in [*drawn, *(ex for _, items, _ in active for ex in items)]),
                  default=1)
    on = decide_checkpointing(model, longest) if checkpointing is None else bool(checkpointing)
    device = model.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokens = 0
    synchronize(device)
    started = time.perf_counter()
    loss_val = 0.0
    with checkpointed(model, on):
        for step in range(start, int(steps) + 1):
            batch = sample_batch(rng, active, active_factories)
            tokens += sum(loss.tokens(ex) for ex in batch)
            loss_val = step_once(model, batch, loss)
            add_to_span(backwards=1)
            opt.update()
            if on_step:
                on_step(step, loss_val)
            if (checkpoint_every and on_checkpoint is not None
                    and step % int(checkpoint_every) == 0 and step < int(steps)):
                on_checkpoint(capture_training_state(params, opt, step, rng))
    synchronize(device)
    seconds = time.perf_counter() - started
    ran = max(0, int(steps) - start + 1)
    return loss_val, {
        "backend": "torch", "accelerator": model.accelerator, "steps": ran, "from_step": start - 1,
        "seconds": round(seconds, 6),
        "seconds_per_step": round(seconds / ran, 6) if ran else None,
        "tokens": tokens, "tokens_per_s": round(tokens / seconds, 2) if seconds > 0 else None,
        "peak_memory_bytes": (int(torch.cuda.max_memory_allocated(device))
                              if device.type == "cuda" else None),
        "gradient_checkpointing": on}
