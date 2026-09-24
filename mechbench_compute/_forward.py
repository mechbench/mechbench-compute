from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.models.gemma4.language import logit_softcap

from . import _arch
from .cache import ActivationCache, kv_offset
from .hooks import HookFn, HookInfo, attn_internal_layers, mlp_internal_layers


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
    shared_kv,
    offset,
    hooks: dict[str, HookFn],
    capture_set: set[str],
    cache: ActivationCache,
    layer_idx: int,
) -> tuple[mx.array, tuple[mx.array, mx.array], mx.array]:
    attn = layer.self_attn
    B, L, _ = x_normed.shape

    queries = attn.q_proj(x_normed).reshape(B, L, attn.n_heads, attn.head_dim)
    queries = _dispatch(
        f"blocks.{layer_idx}.attn.q_pre_norm", layer_idx, "attn.q_pre_norm",
        queries, hooks, capture_set, cache,
    )
    queries = attn.q_norm(queries)

    if shared_kv is not None:
        keys, values = shared_kv
    else:
        offset = mx.array(c.offset) if c is not None else 0
        keys = attn.k_proj(x_normed).reshape(B, L, attn.n_kv_heads, attn.head_dim)
        keys = _dispatch(
            f"blocks.{layer_idx}.attn.k_pre_norm", layer_idx, "attn.k_pre_norm",
            keys, hooks, capture_set, cache,
        )
        values = (
            keys
            if attn.use_k_eq_v
            else attn.v_proj(x_normed).reshape(B, L, attn.n_kv_heads, attn.head_dim)
        )
        keys = attn.k_norm(keys)
        keys = keys.transpose(0, 2, 1, 3)
        keys = _dispatch(
            f"blocks.{layer_idx}.attn.k_pre_rope", layer_idx, "attn.k_pre_rope",
            keys, hooks, capture_set, cache,
        )
        keys = attn.rope(keys, offset=offset)
        values = attn.v_norm(values)
        values = values.transpose(0, 2, 1, 3)
        if c is not None:
            keys, values = c.update_and_fetch(keys, values)

    queries = queries.transpose(0, 2, 1, 3)
    queries = _dispatch(
        f"blocks.{layer_idx}.attn.q_pre_rope", layer_idx, "attn.q_pre_rope",
        queries, hooks, capture_set, cache,
    )
    queries = attn.rope(queries, offset=offset)

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

    if attn.n_heads != attn.n_kv_heads:
        repeats = attn.n_heads // attn.n_kv_heads
        keys_rep = mx.repeat(keys, repeats, axis=1)
        values_rep = mx.repeat(values, repeats, axis=1)
    else:
        keys_rep = keys
        values_rep = values

    scores = (queries @ keys_rep.transpose(0, 1, 3, 2)) * attn.scale

    if mask is not None:
        Q_len = scores.shape[-2]
        K_len = scores.shape[-1]
        if isinstance(mask, mx.array):
            m = mask
            if m.shape[-1] != K_len:
                m = m[..., -K_len:]
            scores = scores + m
        elif mask == "causal":
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

    scores = _dispatch(
        f"blocks.{layer_idx}.attn.scores", layer_idx, "attn.scores", scores,
        hooks, capture_set, cache,
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

    per_head_out = weights @ values_rep
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
    output = _dispatch(
        f"blocks.{layer_idx}.attn.o_in", layer_idx, "attn.o_in", output,
        hooks, capture_set, cache,
    )
    return attn.o_proj(output), (keys, values), offset


# external: mlx-vlm — this mirrors models/gemma4 (gemma4.py Model.__call__, language.py Gemma4TextModel.__call__ and LanguageModel.__call__); calling the model directly skips setup that mlx_vlm.generate performs
def run_forward(
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
    manual_mlp_layer_set = mlp_internal_layers(
        set(hooks.keys()) | capture_set, arch=arch,
    )

    cache = ActivationCache(offset=kv_offset(kv_cache))
    lm = model.language_model
    tm = lm.model

    emb_out = model.get_input_embeddings(input_ids=input_ids, pixel_values=None)
    h = emb_out.inputs_embeds
    h = _dispatch("embed", None, "embed", h, hooks, capture_set, cache)
    per_layer_inputs = emb_out.per_layer_inputs

    if tm.hidden_size_per_layer_input and per_layer_inputs is not None:
        per_layer_inputs = tm.project_per_layer_inputs(h, per_layer_inputs)

    # external: mlx-vlm — make_prompt_cache returns one cache per non-KV-shared layer; shared layers take K/V through previous_kvs
    kv_cache = list(kv_cache if kv_cache is not None else cache_mod.make_prompt_cache(lm))
    kv_cache = kv_cache + [None] * (len(tm.layers) - len(kv_cache))
    masks = tm._make_masks(h, kv_cache, None)
    previous_kvs = tm.previous_kvs
    intermediates: list[tuple] = [(None, None)] * len(tm.layers)

    for i, layer in enumerate(tm.layers):
        if getattr(layer, "enable_moe", False):
            raise NotImplementedError(
                "MoE decoder layers (Gemma 4 26B) are not supported by the "
                "canonical forward. Only dense variants (E2B/E4B/12B) are wired."
            )
        c = kv_cache[i]
        local_mask = masks[i]
        shared_kv, offset = intermediates[previous_kvs[i]]
        per_layer_input = (
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        )

        h = _dispatch(
            f"blocks.{i}.resid_pre", i, "resid_pre", h, hooks, capture_set, cache,
        )
        resid_pre = h

        x_normed = layer.input_layernorm(h)
        x_normed = _dispatch(
            f"blocks.{i}.attn.in_norm", i, "attn.in_norm", x_normed,
            hooks, capture_set, cache,
        )
        if i in manual_attn_layer_set:
            a, new_kv, new_offset = _attention_with_internals(
                layer, x_normed, local_mask, c,
                shared_kv=shared_kv, offset=offset,
                hooks=hooks, capture_set=capture_set, cache=cache, layer_idx=i,
            )
        else:
            a, new_kv, new_offset = layer.self_attn(
                x_normed, local_mask, c, shared_kv=shared_kv, offset=offset,
            )
        intermediates[i] = (new_kv, new_offset)
        a = layer.post_attention_layernorm(a)
        a = _dispatch(
            f"blocks.{i}.attn_out", i, "attn_out", a, hooks, capture_set, cache,
        )
        h = resid_pre + a

        mid = h
        m = layer.pre_feedforward_layernorm(mid)
        m = _dispatch(
            f"blocks.{i}.mlp.in_norm", i, "mlp.in_norm", m,
            hooks, capture_set, cache,
        )
        if i in manual_mlp_layer_set:
            gate = layer.mlp.gate_proj(m)
            gate = _dispatch(
                f"blocks.{i}.mlp.gate", i, "mlp.gate", gate,
                hooks, capture_set, cache,
            )
            up = layer.mlp.up_proj(m)
            up = _dispatch(
                f"blocks.{i}.mlp.up", i, "mlp.up", up,
                hooks, capture_set, cache,
            )
            act = nn.gelu_approx(gate)
            act = _dispatch(
                f"blocks.{i}.mlp.act", i, "mlp.act", act,
                hooks, capture_set, cache,
            )
            down_in = act * up
            down_in = _dispatch(
                f"blocks.{i}.mlp.down_in", i, "mlp.down_in", down_in,
                hooks, capture_set, cache,
            )
            m = layer.mlp.down_proj(down_in)
        else:
            m = layer.mlp(m)
        m = layer.post_feedforward_layernorm(m)
        m = _dispatch(
            f"blocks.{i}.mlp_out", i, "mlp_out", m, hooks, capture_set, cache,
        )
        h = mid + m

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

        if layer.layer_scalar is not None:
            h = h * layer.layer_scalar

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
    h_final = _dispatch("final_norm", None, "final_norm", h_final,
                        hooks, capture_set, cache)
    logits = tm.embed_tokens.as_linear(h_final)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)
    logits = _dispatch("logits", None, "logits", logits,
                       hooks, capture_set, cache)

    mx.eval([logits] + list(cache.values()))
    return logits, cache
