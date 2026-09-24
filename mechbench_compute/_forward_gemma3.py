from __future__ import annotations

import mlx.core as mx
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.models.base import create_attention_mask

from . import _arch
from .cache import ActivationCache, kv_offset
from .hooks import HookFn, HookInfo, attn_internal_layers


def _dispatch(
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


def _attention_with_internals(
    layer,
    x_normed: mx.array,
    mask,
    c,
    *,
    hooks: dict[str, HookFn],
    capture_set: set[str],
    cache: ActivationCache,
    layer_idx: int,
) -> mx.array:
    attn = layer.self_attn
    B, L, _ = x_normed.shape

    q = attn.q_proj(x_normed).reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
    k = attn.k_proj(x_normed).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = attn.v_proj(x_normed).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    q = attn.q_norm(q)
    k = attn.k_norm(k)

    offset = c.offset if c is not None else 0
    q = attn.rope(q, offset=offset)
    k = attn.rope(k, offset=offset)
    if c is not None:
        k, v = c.update_and_fetch(k, v)

    q = _dispatch(
        f"blocks.{layer_idx}.attn.q", layer_idx, "attn.q", q,
        hooks, capture_set, cache,
    )
    k = _dispatch(
        f"blocks.{layer_idx}.attn.k", layer_idx, "attn.k", k,
        hooks, capture_set, cache,
    )
    v = _dispatch(
        f"blocks.{layer_idx}.attn.v", layer_idx, "attn.v", v,
        hooks, capture_set, cache,
    )

    if attn.n_heads != attn.n_kv_heads:
        repeats = attn.n_heads // attn.n_kv_heads
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)

    scores = (q @ k.transpose(0, 1, 3, 2)) * attn.scale

    if mask is not None and isinstance(mask, mx.array):
        m = mask
        if m.shape[-1] != scores.shape[-1]:
            m = m[..., -scores.shape[-1] :]
        scores = scores + m

    weights = mx.softmax(scores, axis=-1)
    weights = _dispatch(
        f"blocks.{layer_idx}.attn.weights", layer_idx, "attn.weights", weights,
        hooks, capture_set, cache,
    )

    per_head_out = weights @ v
    per_head_out = _dispatch(
        f"blocks.{layer_idx}.attn.per_head_out", layer_idx, "attn.per_head_out",
        per_head_out, hooks, capture_set, cache,
    )

    output = per_head_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output)


# external: mlx-vlm — this mirrors models/gemma3/language.py (Gemma3Model, TransformerBlock, Attention, LanguageModel __call__)
def run_forward_gemma3(
    model,
    input_ids: mx.array,
    *,
    hooks: dict[str, HookFn] | None = None,
    capture: list[str] | None = None,
    arch: _arch.Arch | None = None,
    kv_cache=None,
) -> tuple[mx.array, ActivationCache]:
    hooks = dict(hooks or {})
    capture_set = set(capture or [])
    manual_attn_layer_set = attn_internal_layers(
        set(hooks.keys()) | capture_set, arch=arch,
    )

    cache = ActivationCache(offset=kv_offset(kv_cache))
    lm = model.language_model
    tm = lm.model

    h = tm.embed_tokens(input_ids)
    h = h * mx.array(tm.config.hidden_size ** 0.5, mx.bfloat16).astype(h.dtype)

    if kv_cache is None:
        kv_cache = cache_mod.make_prompt_cache(lm)

    pattern = tm.sliding_window_pattern
    global_mask = create_attention_mask(
        h, kv_cache[pattern - 1] if pattern - 1 < len(kv_cache) else None
    )
    sliding_mask = (
        create_attention_mask(h, kv_cache[0], window_size=tm.window_size)
        if pattern > 1
        else None
    )

    for i, layer in enumerate(tm.layers):
        c = kv_cache[i]
        is_global = (i + 1) % pattern == 0
        local_mask = global_mask if is_global else sliding_mask

        h = _dispatch(
            f"blocks.{i}.resid_pre", i, "resid_pre", h, hooks, capture_set, cache,
        )
        resid_pre = h

        x_normed = layer.input_layernorm(h)
        if i in manual_attn_layer_set:
            a = _attention_with_internals(
                layer, x_normed, local_mask, c,
                hooks=hooks, capture_set=capture_set, cache=cache, layer_idx=i,
            )
        else:
            a = layer.self_attn(x_normed, local_mask, c)
        a = layer.post_attention_layernorm(a)
        a = _dispatch(
            f"blocks.{i}.attn_out", i, "attn_out", a, hooks, capture_set, cache,
        )
        h = resid_pre + a

        mid = h
        m_in = layer.pre_feedforward_layernorm(mid)
        m_out = layer.mlp(m_in)
        m_out = layer.post_feedforward_layernorm(m_out)
        m_out = _dispatch(
            f"blocks.{i}.mlp_out", i, "mlp_out", m_out, hooks, capture_set, cache,
        )
        h = mid + m_out

        h = _dispatch(
            f"blocks.{i}.resid_post", i, "resid_post", h, hooks, capture_set, cache,
        )

    if "final_norm.scale" in capture_set or "final_norm.scale" in hooks:
        f32 = h.astype(mx.float32)
        eps = float(getattr(tm.norm, "eps", 1e-6))
        rms = mx.sqrt(mx.mean(f32 * f32, axis=-1) + eps)
        _dispatch("final_norm.scale", None, "final_norm.scale", rms,
                  hooks, capture_set, cache)

    h_final = tm.norm(h)
    logits = lm.lm_head(h_final)

    mx.eval([logits] + list(cache.values()))
    return logits, cache
