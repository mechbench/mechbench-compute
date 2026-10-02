from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from mechbench_compute.support import Unembed


def cap_logits(unembed: Unembed, logits: mx.array) -> mx.array:
    if unembed.softcap is None:
        return logits
    from mlx_vlm.models.gemma4.language import logit_softcap

    return logit_softcap(unembed.softcap, logits)


def make_head_logits(read_unembed: Callable[[Any], Unembed]) -> Callable[[Any, mx.array], mx.array]:
    def head_logits(model, hidden: mx.array) -> mx.array:
        u = read_unembed(model)
        return cap_logits(u, u.project(hidden))

    return head_logits


def make_project_to_logits(read_unembed: Callable[[Any], Unembed]) -> Callable[[Any, mx.array], mx.array]:
    def project_to_logits(model, residual: mx.array) -> mx.array:
        u = read_unembed(model)
        return cap_logits(u, u.project(u.norm(residual)))

    return project_to_logits
