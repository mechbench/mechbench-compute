from __future__ import annotations

from typing import Any

from mlx.utils import tree_flatten

from mechbench_compute.adapters.attach_operators import (
    attach_operators,
    detach_operators,
)
from mechbench_compute.adapters.build_operator_modules import build_operator_modules
from mechbench_compute.adapters.is_operator import KIND
from mechbench_compute.adapters.prune_operators import prune_operators
from mechbench_compute.adapters.read_operator_spec import read_operator_spec
from mechbench_compute.finetune import train_soft_ce


def train_operator(ctx, model, operator: Any, groups, batch, *, trained_on, methods,
                   checkpoint_every: int) -> dict[str, Any]:
    lm, d = model.lm, int(model.arch.d_model)
    spec = read_operator_spec(operator, n_layers=len(lm.model.layers), d=d)
    steps, seed = int(methods["steps"]), int(methods["seed"])
    modules = build_operator_modules(spec, d, seed=seed)
    l1 = (spec.get("penalty") or {}).get("l1")
    lm.freeze()
    handle = attach_operators(lm, modules)
    try:
        if ctx.on_start:
            ctx.on_start(steps)
        resumed_from = int(ctx.resume_state["step"]) if ctx.resume_state else 0
        for _ in range(resumed_from if ctx.on_item else 0):
            ctx.on_item(None, None, True)
        final_loss = train_soft_ce(
            lm, groups, batch, steps=steps, lr=float(methods["lr"]), seed=seed,
            on_step=(lambda step, loss: ctx.on_item()) if ctx.on_item else None,
            checkpoint_every=checkpoint_every if ctx.on_checkpoint else 0,
            on_checkpoint=ctx.on_checkpoint, resume_state=ctx.resume_state,
            after_update=(lambda opt: prune_operators(lm, handle, opt, l1)) if l1 else None)
        values = {str(i): m.read_values() for i, m in modules.items()}
        nominal = sum(v.size for m in modules.values() for _, v in tree_flatten(m.parameters()))
        effective = sum(m.count_effective() for m in modules.values())
    finally:
        detach_operators(lm, handle)
    return {"kind": KIND, "base_model": trained_on["base"], "trained_on": trained_on,
            "operator": {**spec, "d": d, "params": nominal, "effective": effective},
            "parameters": values, "train": {**methods, "final_loss": round(final_loss, 4)}}
