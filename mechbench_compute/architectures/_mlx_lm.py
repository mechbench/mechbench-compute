from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx import nn

from mechbench_compute._arch import Arch
from mechbench_compute._attention_mask import apply_mask
from mechbench_compute.architectures._dispatch import (
    dispatch,
    read_layers,
    run_head,
    run_mlp_with_internals,
    stop_early,
)
from mechbench_compute.cache import ActivationCache, kv_offset
from mechbench_compute.head_weights import make_head_weights
from mechbench_compute.hooks import HookFn, attn_internal_layers, mlp_internal_layers
from mechbench_compute.support import Unembed


def load_lm(model_id: str, **config: Any) -> tuple[Any, Any]:
    from mlx_lm import load

    return load(model_id, **config)


def read_lm(model):
    return model


read_lm_head_weights = make_head_weights(
    lambda model: model.model.layers,
    read_n_heads=lambda attn: attn.n_heads,
    is_global=lambda layer: not getattr(layer, "use_sliding", False))


def make_lm_cache(model):
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def tokenize_lm(model, processor, prompt: str, *, chat_template: bool = True) -> mx.array:
    if chat_template:
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt
    ids = processor.encode(rendered)
    return mx.array([ids], dtype=mx.int32)


def read_unembed(model) -> Unembed:
    tm = model.model
    project = tm.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    return Unembed(norm=tm.norm, project=project)


def read_lm_arch(model_type: str, model, model_id: str | None = None) -> Arch:
    args = model.args
    n_layers = int(args.num_hidden_layers)
    layer_types = getattr(args, "layer_types", None)
    if layer_types:
        global_layers = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    else:
        global_layers = tuple(range(n_layers))
    return Arch(
        model_id=model_id or "",
        n_layers=n_layers,
        d_model=int(args.hidden_size),
        n_heads=int(args.num_attention_heads),
        n_kv_heads=int(args.num_key_value_heads),
        vocab_size=int(args.vocab_size),
        hidden_size_per_layer_input=0,
        global_layers=global_layers,
        first_kv_shared_layer=n_layers,
        model_type=model_type,
    )


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


# external: mlx-lm — this mirrors models/llama.py and models/qwen2.py (Model, TransformerBlock, Attention __call__)
def run_lm_forward(
    model,
    input_ids: mx.array,
    *,
    make_masks: Callable[[Any, mx.array, list], list],
    hooks: dict[str, HookFn] | None = None,
    capture: list[str] | None = None,
    arch: Arch | None = None,
    kv_cache=None,
    stop_after: int | None = None,
) -> tuple[mx.array | None, ActivationCache]:
    hooks = dict(hooks or {})
    capture_set = set(capture or [])
    manual_attn_layer_set = attn_internal_layers(
        set(hooks.keys()) | capture_set, arch=arch,
    )
    manual_mlp_layer_set = mlp_internal_layers(
        set(hooks.keys()) | capture_set, arch=arch,
    )

    cache = ActivationCache(offset=kv_offset(kv_cache))
    tm = model.model

    h = tm.embed_tokens(input_ids)
    h = dispatch("embed", None, "embed", h, hooks, capture_set, cache)
    if kv_cache is None:
        kv_cache = make_lm_cache(model)
    masks = make_masks(tm, h, kv_cache)

    for i, layer in enumerate(read_layers(tm.layers, stop_after)):
        c = kv_cache[i]
        local_mask = masks[i]

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
        a = dispatch(
            f"blocks.{i}.attn_out", i, "attn_out", a, hooks, capture_set, cache,
        )
        h = resid_pre + a

        mid = h
        m_in = layer.post_attention_layernorm(mid)
        m_in = dispatch(
            f"blocks.{i}.mlp.in_norm", i, "mlp.in_norm", m_in, hooks, capture_set, cache,
        )
        if i in manual_mlp_layer_set:
            m_out = run_mlp_with_internals(
                layer.mlp, m_in, act=nn.silu,
                hooks=hooks, capture_set=capture_set, cache=cache, layer_idx=i,
            )
        else:
            m_out = layer.mlp(m_in)
        m_out = dispatch(
            f"blocks.{i}.mlp_out", i, "mlp_out", m_out, hooks, capture_set, cache,
        )
        h = mid + m_out

        h = dispatch(
            f"blocks.{i}.resid_post", i, "resid_post", h, hooks, capture_set, cache,
        )

    if stop_after is not None:
        return stop_early(cache)
    logits = run_head(h, norm=tm.norm, unembed=read_unembed(model).project,
                      hooks=hooks, capture_set=capture_set, cache=cache)
    return logits, cache
