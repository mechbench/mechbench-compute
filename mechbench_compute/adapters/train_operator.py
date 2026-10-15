from __future__ import annotations

from typing import Any

from mechbench_compute.adapters.is_operator import KIND
from mechbench_compute.adapters.read_operator_spec import read_operator_spec
from mechbench_compute.backends import backend_of
from mechbench_compute.distill import soft_ce


def train_operator(ctx, model, operator: Any, groups, batch, *, trained_on, methods,
                   checkpoint_every: int, deterministic: bool = True) -> dict[str, Any]:
    lm, d = model.lm, int(model.arch.d_model)
    spec = read_operator_spec(operator, n_layers=len(lm.model.layers), d=d)
    if backend_of(model) == "torch":
        from mechbench_compute.torch_backend.determinism import read_training_numerics

        methods = {**methods, "numerics": read_training_numerics(model.device, deterministic)}
        final_loss, values, nominal, effective = train_on_torch(
            ctx, model, spec, d, groups, batch, methods, checkpoint_every, deterministic)
    else:
        final_loss, values, nominal, effective = train_on_mlx(
            ctx, model, spec, d, groups, batch, methods, checkpoint_every)
    return {"kind": KIND, "base_model": trained_on["base"], "trained_on": trained_on,
            "operator": {**spec, "d": d, "params": nominal, "effective": effective},
            "parameters": values, "train": {**methods, "final_loss": round(final_loss, 4)}}


def read_loop(ctx, methods, checkpoint_every: int) -> dict[str, Any]:
    steps = int(methods["steps"])
    if ctx.on_start:
        ctx.on_start(steps)
    resumed_from = int(ctx.resume_state["step"]) if ctx.resume_state else 0
    for _ in range(resumed_from if ctx.on_item else 0):
        ctx.on_item(None, None, True)
    return {"steps": steps, "lr": float(methods["lr"]), "seed": int(methods["seed"]),
            "on_step": (lambda step, loss: ctx.on_item()) if ctx.on_item else None,
            "checkpoint_every": checkpoint_every if ctx.on_checkpoint else 0,
            "on_checkpoint": ctx.on_checkpoint, "resume_state": ctx.resume_state}


def train_on_mlx(ctx, model, spec, d, groups, batch, methods, checkpoint_every):
    from mlx.utils import tree_flatten

    from mechbench_compute.adapters.attach_operators import (
        attach_operators,
        detach_operators,
    )
    from mechbench_compute.adapters.build_operator_modules import build_operator_modules
    from mechbench_compute.adapters.prune_operators import prune_operators
    from mechbench_compute.finetune import train_soft_ce

    lm = model.lm
    modules = build_operator_modules(spec, d, seed=int(methods["seed"]))
    l1 = (spec.get("penalty") or {}).get("l1")
    lm.freeze()
    handle = attach_operators(lm, modules)
    try:
        final_loss = train_soft_ce(
            lm, groups, batch, **read_loop(ctx, methods, checkpoint_every),
            after_update=(lambda opt: prune_operators(lm, handle, opt, l1)) if l1 else None)
        values = {str(i): m.read_values() for i, m in modules.items()}
        nominal = sum(v.size for m in modules.values() for _, v in tree_flatten(m.parameters()))
        effective = sum(m.count_effective() for m in modules.values())
    finally:
        detach_operators(lm, handle)
    return final_loss, values, nominal, effective


def train_on_torch(ctx, model, spec, d, groups, batch, methods, checkpoint_every, deterministic):
    from mechbench_compute.torch_backend import operators as ops
    from mechbench_compute.torch_backend.training import train_soft_ce

    built = ops.build_operators(spec, d, seed=int(methods["seed"]), device=model.device)
    l1 = (spec.get("penalty") or {}).get("l1")
    held = {name: p.requires_grad for name, p in model._model.named_parameters()}
    for p in model._model.parameters():
        p.requires_grad_(False)
    handle = ops.attach_operators(model.lm, built)
    try:
        final_loss = train_soft_ce(
            model, groups, batch, **read_loop(ctx, methods, checkpoint_every), loss_fn=soft_ce,
            deterministic=deterministic, parameters=ops.read_operator_parameters(built),
            after_update=(lambda opt: ops.prune_operators(built, opt, l1)) if l1 else None)
        values = {str(i): op.read_values() for i, op in built.items()}
        nominal = sum(p.numel() for op in built.values() for p in op.params.values())
        effective = sum(op.count_effective() for op in built.values())
    finally:
        ops.detach_operators(handle)
        for name, p in model._model.named_parameters():
            p.requires_grad_(held.get(name, p.requires_grad))
    return final_loss, values, nominal, effective
