"""Canonical hook-aware forward pass for Llama 3 / 3.1 / 3.2.

Counterpart to `_forward_qwen.py` for the Llama 3 family loaded
via mlx-lm. Mirrors:

  - mlx_lm/models/llama.py LlamaModel.__call__ (the layer loop +
    per-layer mask routing for hybrid attention)
  - mlx_lm/models/llama.py TransformerBlock.__call__
  - mlx_lm/models/llama.py Attention.__call__

Key differences from Qwen 2:

  - **Unbiased Q/K/V projections** by default (Qwen has bias).
    Llama 3's `args.attention_bias` is configurable but typically
    False; the manual-attention path doesn't care either way.
  - **Per-layer hybrid attention** via `args.layer_types`. Some
    Llama 3 variants are pure full-attention (e.g. Llama 3.1 8B);
    others may have sliding-window layers (Llama 3.2 1B / 3B
    inherit the hybrid pattern from upstream). `LlamaModel`
    constructs both `fa_mask` and `swa_mask` per forward and
    routes per-layer.
  - **Tied or untied unembed** depending on `args.tie_word_embeddings`.

The TransformerBlock structure is identical to Qwen 2's:
input_layernorm → attention → +residual; post_attention_layernorm
→ MLP → +residual. The forward is essentially `_forward_qwen.py`
with per-layer mask routing added.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache

from . import _arch
from .cache import ActivationCache
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
        info = HookInfo(name=name, layer=layer, point=point)
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
    """Manually computed Llama 3 attention exposing weights and per-head
    output. Mirrors mlx_lm/models/llama.py Attention.__call__ structurally;
    matches Qwen 2's manual path because the attention math is identical
    (the only Qwen-vs-Llama-3 difference is q/k/v_proj bias, which the
    Linear layer handles internally)."""
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


def run_forward_llama(
    model,
    input_ids: mx.array,
    *,
    hooks: dict[str, HookFn] | None = None,
    capture: list[str] | None = None,
    arch: _arch.Arch | None = None,
) -> tuple[mx.array, ActivationCache]:
    """Run a single hook-aware forward pass through a Llama 3 model
    loaded via mlx-lm."""
    hooks = dict(hooks or {})
    capture_set = set(capture or [])
    manual_attn_layer_set = attn_internal_layers(
        set(hooks.keys()) | capture_set, arch=arch,
    )

    cache = ActivationCache()
    tm = model.model  # LlamaModel

    h = tm.embed_tokens(input_ids)
    kv_cache = make_prompt_cache(model)

    # Per-layer hybrid attention masks (Llama 3.2 may use sliding window;
    # 3.1 8B is full-attention only, in which case swa_idx is None and
    # swa_mask isn't constructed).
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

    h_final = tm.norm(h)
    if model.args.tie_word_embeddings:
        logits = tm.embed_tokens.as_linear(h_final)
    else:
        logits = model.lm_head(h_final)

    mx.eval([logits] + list(cache.values()))
    return logits, cache
