from __future__ import annotations

import mlx.core as mx
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.models.base import create_attention_mask

from mechbench_compute._arch import Arch
from mechbench_compute._attention_mask import apply_mask
from mechbench_compute.architectures._dispatch import dispatch, run_head
from mechbench_compute.architectures._head import (
    make_head_logits,
    make_project_to_logits,
    refuse_head_weights,
)
from mechbench_compute.architectures._vlm import (
    load_vlm,
    make_vlm_cache,
    read_language_model,
    read_vlm_arch,
    tokenize_vlm,
)
from mechbench_compute.cache import ActivationCache, kv_offset
from mechbench_compute.hooks import HookFn, attn_internal_layers
from mechbench_compute.lora import ADAPTER_KEYS
from mechbench_compute.support import (
    CORE_GLOBAL_POINTS,
    CORE_LAYER_POINTS,
    Architecture,
    Unembed,
    refuse_logit_softcap,
)


def read_unembed(model) -> Unembed:
    lm = model.language_model
    return Unembed(norm=lm.model.norm, project=lm.lm_head)


def read_arch(model, model_id: str | None = None) -> Arch:
    return read_vlm_arch("gemma3", model.config.text_config, model_id)


def read_attn_out_norm(model, layer: int):
    return model.language_model.model.layers[layer].post_attention_layernorm


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

    q = dispatch(
        f"blocks.{layer_idx}.attn.q", layer_idx, "attn.q", q,
        hooks, capture_set, cache,
    )
    k = dispatch(
        f"blocks.{layer_idx}.attn.k", layer_idx, "attn.k", k,
        hooks, capture_set, cache,
    )
    v = dispatch(
        f"blocks.{layer_idx}.attn.v", layer_idx, "attn.v", v,
        hooks, capture_set, cache,
    )

    if attn.n_heads != attn.n_kv_heads:
        repeats = attn.n_heads // attn.n_kv_heads
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)

    scores = (q @ k.transpose(0, 1, 3, 2)) * attn.scale

    scores = apply_mask(scores, mask)

    weights = mx.softmax(scores, axis=-1)
    weights = dispatch(
        f"blocks.{layer_idx}.attn.weights", layer_idx, "attn.weights", weights,
        hooks, capture_set, cache,
    )

    per_head_out = weights @ v
    per_head_out = dispatch(
        f"blocks.{layer_idx}.attn.per_head_out", layer_idx, "attn.per_head_out",
        per_head_out, hooks, capture_set, cache,
    )

    output = per_head_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output)


# external: mlx-vlm — this mirrors models/gemma3/language.py (Gemma3Model, TransformerBlock, Attention, LanguageModel __call__)
def run_forward(
    model,
    input_ids: mx.array,
    *,
    hooks: dict[str, HookFn] | None = None,
    capture: list[str] | None = None,
    arch: Arch | None = None,
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
    h = dispatch("embed", None, "embed", h, hooks, capture_set, cache)

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

        h = dispatch(
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
        a = dispatch(
            f"blocks.{i}.attn_out", i, "attn_out", a, hooks, capture_set, cache,
        )
        h = resid_pre + a

        mid = h
        m_in = layer.pre_feedforward_layernorm(mid)
        m_out = layer.mlp(m_in)
        m_out = layer.post_feedforward_layernorm(m_out)
        m_out = dispatch(
            f"blocks.{i}.mlp_out", i, "mlp_out", m_out, hooks, capture_set, cache,
        )
        h = mid + m_out

        h = dispatch(
            f"blocks.{i}.resid_post", i, "resid_post", h, hooks, capture_set, cache,
        )

    logits = run_head(h, norm=tm.norm, unembed=lm.lm_head,
                      hooks=hooks, capture_set=capture_set, cache=cache)
    return logits, cache


ARCH = Architecture(
    model_type="gemma3", name="Gemma 3", loader="mlx-vlm",
    generate=True, score=True, train=True,
    layer_points=CORE_LAYER_POINTS, global_points=CORE_GLOBAL_POINTS,
    residual_law="resid_post[i] == resid_pre[i] + attn_out[i] + mlp_out[i] == resid_pre[i+1]",
    load=load_vlm,
    arch_of=read_arch,
    forward=run_forward,
    lm=read_language_model,
    prompt_cache=make_vlm_cache,
    head_logits=make_head_logits(read_unembed),
    project_to_logits=make_project_to_logits(read_unembed),
    tokenize=tokenize_vlm,
    attribution_unembed=read_unembed,
    head_weights=refuse_head_weights("gemma3"),
    dialect=None,
    reasoning=(),
    adapter_keys=ADAPTER_KEYS,
    refused_when=(refuse_logit_softcap("Gemma 3"),),
    config_defaults={
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
        "vocab_size": 262208, "sliding_window_pattern": 6,
    },
    attn_out_norm=read_attn_out_norm,
)
