"""Canonical hook-aware forward pass for Gemma 4 E4B.

This module owns the model's layer loop. There is one and only one path
through the network. Different experimental needs are expressed by passing
different hooks/captures, never by calling a different function — the bug
that motivates this design is documented in CLAUDE.md ("Known open bug"),
where calling model(input_ids) directly produces garbage because it skips
setup that mlx_vlm.generate performs. Every existing experiment script
worked around it by reimplementing the layer loop inline. This module
consolidates those reimplementations into the single source of truth.

The forward pass mirrors:
  - mlx_vlm/utils.py prepare_inputs (handled by Model.tokenize, not here)
  - mlx_vlm/models/gemma4/gemma4.py Model.__call__ (embeddings + per-layer)
  - mlx_vlm/models/gemma4/language.py Gemma4TextModel.__call__ (the layer loop)
  - mlx_vlm/models/gemma4/language.py LanguageModel.__call__ (norm + unembed)

Two attention paths are supported. The fused path uses MLX's
scaled_dot_product_attention kernel (faster). The manual path computes
Q @ K^T / softmax / @ V explicitly so the attention weights are observable
(needed only when a hook/capture targets attn.weights or attn.per_head_out).
The path is chosen automatically per call.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.gemma4.language import logit_softcap

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
    """Invoke the user hook (if any) at this point, then capture (if requested),
    and return the activation that should flow forward. Captures see the
    post-hook value, so an ablation hook + capture at the same point records
    the ablated state (which is what you want for assertions).
    """
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
    """Manually computed attention exposing weights and per-head output.

    Mirrors Attention.__call__ in mlx_vlm/models/gemma4/language.py exactly,
    but replaces the fused scaled_dot_product_attention with a manual softmax
    so the post-softmax weights are inspectable. Keep this function in lock-
    step with the upstream Attention implementation if mlx-vlm ever changes.
    """
    attn = layer.self_attn
    B, L, _ = x_normed.shape

    queries = attn.q_proj(x_normed).reshape(B, L, attn.n_heads, attn.head_dim)
    queries = attn.q_norm(queries)

    offset = 0
    if attn.is_kv_shared_layer and c is not None:
        # Shared layer: read keys/values from the cache populated by an
        # earlier non-shared layer. Don't recompute or rewrite the cache.
        state = c.state
        keys, values = state[0], state[1]
        offset = c.offset
    else:
        if c is not None:
            offset = c.offset
        keys = attn.k_proj(x_normed).reshape(B, L, attn.n_kv_heads, attn.head_dim)
        values = (
            keys
            if attn.use_k_eq_v
            else attn.v_proj(x_normed).reshape(B, L, attn.n_kv_heads, attn.head_dim)
        )
        keys = attn.k_norm(keys)
        values = attn.v_norm(values)
        values = values.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        keys = attn.rope(keys, offset=offset)
        if c is not None:
            keys, values = c.update_and_fetch(keys, values)

    queries = queries.transpose(0, 2, 1, 3)
    queries = attn.rope(queries, offset=offset)

    # Expose Q, K, V as hook points BEFORE the GQA-repeat. Users who want
    # post-repeat copies can mx.repeat themselves; the natural per-KV-head
    # shape is more interpretable as the trained storage.
    queries = _dispatch(
        f"blocks.{layer_idx}.attn.q", layer_idx, "attn.q", queries,
        hooks, capture_set, cache,
    )
    keys = _dispatch(
        f"blocks.{layer_idx}.attn.k", layer_idx, "attn.k", keys,
        hooks, capture_set, cache,
    )
    values = _dispatch(
        f"blocks.{layer_idx}.attn.v", layer_idx, "attn.v", values,
        hooks, capture_set, cache,
    )

    # Grouped-query attention: repeat KV heads to match query heads.
    if attn.n_heads != attn.n_kv_heads:
        repeats = attn.n_heads // attn.n_kv_heads
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)

    scores = (queries @ keys.transpose(0, 1, 3, 2)) * attn.scale

    # Apply attention mask. create_attention_mask can return None, a string
    # ('causal'), or an mx.array. The fused scaled_dot_product_attention
    # handles the string internally; we have to materialize it explicitly
    # here. The previous implementation only handled mx.array masks and
    # silently produced NO-MASK (bidirectional) attention whenever the
    # framework returned 'causal' — a latent bug in step_05/06/07 that
    # happened not to bite those experiments because they only looked at
    # attention FROM the final token position (which attends to everything
    # under causal anyway).
    if mask is not None:
        Q_len = scores.shape[-2]
        K_len = scores.shape[-1]
        if isinstance(mask, mx.array):
            m = mask
            if m.shape[-1] != K_len:
                m = m[..., -K_len:]
            scores = scores + m
        elif mask == "causal":
            # Build [Q_len, K_len] causal mask. K_len may exceed Q_len when
            # the KV cache holds a prefix (K_len == offset + Q_len). Allow
            # query i to attend to key j whenever j <= (K_len - Q_len) + i.
            i = mx.arange(Q_len).reshape(Q_len, 1)
            j = mx.arange(K_len).reshape(1, K_len)
            allowed = j <= (K_len - Q_len + i)
            m = mx.where(allowed, mx.array(0.0, dtype=scores.dtype),
                          mx.array(-1e9, dtype=scores.dtype))
            scores = scores + m
        else:
            raise NotImplementedError(
                f"Unknown mask type in manual attention path: {mask!r}. "
                f"Expected None, 'causal', or an mx.array."
            )

    weights = mx.softmax(scores, axis=-1)
    weights = _dispatch(
        f"blocks.{layer_idx}.attn.weights",
        layer_idx,
        "attn.weights",
        weights,
        hooks,
        capture_set,
        cache,
    )

    per_head_out = weights @ values  # [B, n_heads, L, head_dim]
    per_head_out = _dispatch(
        f"blocks.{layer_idx}.attn.per_head_out",
        layer_idx,
        "attn.per_head_out",
        per_head_out,
        hooks,
        capture_set,
        cache,
    )

    output = per_head_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output)


def run_forward(
    model,
    input_ids: mx.array,
    *,
    hooks: dict[str, HookFn] | None = None,
    capture: list[str] | None = None,
    arch: _arch.Arch | None = None,
) -> tuple[mx.array, ActivationCache]:
    """Run a single forward pass through Gemma 4 E4B.

    Returns (logits, cache). Logits are [1, seq_len, vocab_size] in bf16.
    Cache contains exactly the activations named in `capture`, each
    materialized via mx.eval before return.

    All hook-name validation is the caller's responsibility (Model.run does
    it). This function trusts the names it receives.
    """
    hooks = dict(hooks or {})
    capture_set = set(capture or [])

    # Per-layer attention path selection. Manual softmax is slower than the
    # fused SDPA kernel and produces slightly different bf16 rounding, so we
    # only use it at layers where the user wants to inspect attention
    # internals. Other layers stay on the fused path. This keeps the residual
    # stream bitwise-equivalent with mlx_vlm's standard forward at every
    # layer the user isn't actively probing.
    manual_attn_layer_set = attn_internal_layers(
        set(hooks.keys()) | capture_set, arch=arch,
    )

    cache = ActivationCache()
    lm = model.language_model
    tm = lm.model  # Gemma4TextModel

    # ---- Embeddings + MatFormer per-layer-input side-channel ----
    emb_out = model.get_input_embeddings(input_ids=input_ids, pixel_values=None)
    h = emb_out.inputs_embeds
    per_layer_inputs = emb_out.per_layer_inputs

    if tm.hidden_size_per_layer_input and per_layer_inputs is not None:
        per_layer_inputs = tm.project_per_layer_inputs(h, per_layer_inputs)

    # ---- KV cache + hybrid attention masks ----
    kv_cache = cache_mod.make_prompt_cache(lm)
    global_mask = create_attention_mask(
        h,
        kv_cache[tm.first_full_cache_idx]
        if tm.first_full_cache_idx < len(kv_cache)
        else None,
    )
    sliding_mask = create_attention_mask(
        h,
        kv_cache[tm.first_sliding_cache_idx]
        if tm.first_sliding_cache_idx < len(kv_cache)
        else None,
        window_size=tm.window_size,
    )

    for i, layer in enumerate(tm.layers):
        c = kv_cache[tm.layer_idx_to_cache_idx[i]]
        is_global = layer.layer_type == "full_attention"
        local_mask = global_mask if is_global else sliding_mask
        per_layer_input = (
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        )

        # ---- resid_pre ----
        h = _dispatch(
            f"blocks.{i}.resid_pre", i, "resid_pre", h, hooks, capture_set, cache,
        )
        resid_pre = h

        # ---- Attention branch ----
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

        # ---- MLP branch ----
        mid = h
        m = layer.pre_feedforward_layernorm(mid)
        m = layer.mlp(m)
        m = layer.post_feedforward_layernorm(m)
        m = _dispatch(
            f"blocks.{i}.mlp_out", i, "mlp_out", m, hooks, capture_set, cache,
        )
        h = mid + m

        # ---- Per-layer-input side-channel (the MatFormer gate) ----
        if (
            layer.per_layer_input_gate is not None
            and layer.per_layer_projection is not None
            and layer.post_per_layer_input_norm is not None
            and per_layer_input is not None
        ):
            gate = layer.per_layer_input_gate(h)
            gate = nn.gelu_approx(gate)
            gate = mx.multiply(gate, per_layer_input)
            gate = layer.per_layer_projection(gate)
            gate = layer.post_per_layer_input_norm(gate)
            gate = _dispatch(
                f"blocks.{i}.gate_out", i, "gate_out", gate,
                hooks, capture_set, cache,
            )
            h = h + gate

        # ---- Layer scalar ----
        if layer.layer_scalar is not None:
            h = h * layer.layer_scalar

        h = _dispatch(
            f"blocks.{i}.resid_post", i, "resid_post", h, hooks, capture_set, cache,
        )

    # ---- Final norm + tied unembed (+ optional softcap) ----
    h_final = tm.norm(h)
    logits = tm.embed_tokens.as_linear(h_final)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)

    # Single batched eval. MLX is lazy; until we eval, the cache holds graph
    # nodes rather than computed tensors.
    mx.eval([logits] + list(cache.values()))
    return logits, cache
