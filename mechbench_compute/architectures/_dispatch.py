from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from mechbench_compute.cache import ActivationCache
from mechbench_compute.hooks import HookFn, dispatch


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


def run_mlp_with_internals(
    mlp,
    x_normed: mx.array,
    *,
    act: Callable[[mx.array], mx.array],
    hooks: dict[str, HookFn],
    capture_set: set[str],
    cache: ActivationCache,
    layer_idx: int,
) -> mx.array:
    named = f"blocks.{layer_idx}.mlp."
    gate = dispatch(named + "gate", layer_idx, "mlp.gate", mlp.gate_proj(x_normed),
                    hooks, capture_set, cache)
    up = dispatch(named + "up", layer_idx, "mlp.up", mlp.up_proj(x_normed),
                  hooks, capture_set, cache)
    a = dispatch(named + "act", layer_idx, "mlp.act", act(gate), hooks, capture_set, cache)
    down_in = dispatch(named + "down_in", layer_idx, "mlp.down_in", a * up,
                       hooks, capture_set, cache)
    return mlp.down_proj(down_in)


def read_layers(layers, stop_after: int | None):
    return layers if stop_after is None else layers[:stop_after + 1]


def stop_early(cache: ActivationCache) -> tuple[None, ActivationCache]:
    mx.eval(list(cache.values()))
    return None, cache
