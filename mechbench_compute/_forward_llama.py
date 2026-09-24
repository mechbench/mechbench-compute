from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache

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


# external: mlx-lm — this mirrors models/llama.py (LlamaModel, TransformerBlock, Attention __call__)
def run_forward_llama(
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
    tm = model.model

    h = tm.embed_tokens(input_ids)
    if kv_cache is None:
        kv_cache = make_prompt_cache(model)

    fa_mask = create_attention_mask(h, kv_cache[tm.fa_idx])
    swa_mask = None
    if tm.swa_idx is not None:
        swa_mask = create_attention_mask(
            h, kv_cache[tm.swa_idx], window_size=tm.sliding_window
        )

    for i, layer in enumerate(tm.layers):
        c = kv_cache[i]
        local_mask = swa_mask if layer.use_sliding else fa_mask

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
        a = _dispatch(
            f"blocks.{i}.attn_out", i, "attn_out", a, hooks, capture_set, cache,
        )
        h = resid_pre + a

        mid = h
        m_in = layer.post_attention_layernorm(mid)
        m_out = layer.mlp(m_in)
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
    if model.args.tie_word_embeddings:
        logits = tm.embed_tokens.as_linear(h_final)
    else:
        logits = model.lm_head(h_final)

    mx.eval([logits] + list(cache.values()))
    return logits, cache
