from __future__ import annotations

from typing import Any

import mlx.core as mx

from mechbench_compute.adapters.attach_operators import SLOT, OperatorHandle


def prune_operators(lm, handle: OperatorHandle, opt: Any, l1: float) -> None:
    lr = float(opt.learning_rate)
    for i, module in handle.attached:
        held = lm.model.layers[i][SLOT]
        j = next(n for n, m in enumerate(held) if m is module)
        state = opt.state["model"]["layers"][i][SLOT][j]
        for name, centre in module.read_identity().items():
            value = getattr(module, name)
            room = l1 * lr / (mx.sqrt(state[name]["v"]) + opt.eps)
            moved = value - centre
            setattr(module, name, centre + mx.sign(moved) * mx.maximum(mx.abs(moved) - room, 0.0))
