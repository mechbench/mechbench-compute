from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from mechbench_compute.cache import ActivationCache
from mechbench_compute.hooks import HookFn, HookInfo


def dispatch(
    name: str,
    layer: int | None,
    point: str,
    activation: mx.array,
    hooks: dict[str, HookFn],
    capture_set: set[str],
    cache: ActivationCache,
) -> mx.array:
    fn = hooks.get(name)
    if fn is not None:
        info = HookInfo(name=name, layer=layer, point=point, offset=cache.offset)
        new = fn(activation, info)
        if new is not None:
            activation = new
    if name in capture_set:
        cache[name] = activation
    return activation


def run_head(
    h: mx.array,
    *,
    norm,
    unembed: Callable[[mx.array], mx.array],
    hooks: dict[str, HookFn],
    capture_set: set[str],
    cache: ActivationCache,
) -> mx.array:
    if "final_norm.scale" in capture_set or "final_norm.scale" in hooks:
        f32 = h.astype(mx.float32)
        eps = float(getattr(norm, "eps", 1e-6))
        rms = mx.sqrt(mx.mean(f32 * f32, axis=-1) + eps)
        dispatch("final_norm.scale", None, "final_norm.scale", rms,
                 hooks, capture_set, cache)
    h_final = norm(h)
    h_final = dispatch("final_norm", None, "final_norm", h_final,
                       hooks, capture_set, cache)
    logits = unembed(h_final)
    logits = dispatch("logits", None, "logits", logits, hooks, capture_set, cache)
    mx.eval([logits] + list(cache.values()))
    return logits
