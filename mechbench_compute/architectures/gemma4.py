from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models import cache as cache_mod

from mechbench_compute._arch import GLOBAL_HOOK_POINTS, LAYER_HOOK_POINTS, Arch
from mechbench_compute._attention_mask import apply_mask
from mechbench_compute.adapter_keys import ADAPTER_KEYS
from mechbench_compute.architectures._dispatch import dispatch, run_head
from mechbench_compute.architectures._head import (
    cap_logits,
    make_head_logits,
    make_project_to_logits,
)
from mechbench_compute.architectures._vlm import (
    load_vlm,
    make_vlm_cache,
    read_language_model,
    read_vlm_arch,
    tokenize_vlm,
)
from mechbench_compute.cache import ActivationCache, kv_offset
from mechbench_compute.head_weights import HeadSpec, read_dense_weight
from mechbench_compute.hooks import HookFn, attn_internal_layers, mlp_internal_layers
from mechbench_compute.support import Absence, Architecture, Refusal, Unembed
from mechbench_compute.thinking import Delimiters
from mechbench_compute.tool_dialects.gemma4 import DIALECT


def read_unembed(model) -> Unembed:
    lm = model.language_model
    return Unembed(norm=lm.model.norm, project=lm.model.embed_tokens.as_linear,
                   softcap=lm.final_logit_softcapping)


def read_arch(model, model_id: str | None = None) -> Arch:
    return read_vlm_arch("gemma4", model.config.text_config, model_id)


def read_layer_scalars(model) -> tuple[float, ...]:
    return tuple(1.0 if layer.layer_scalar is None
                 else float(layer.layer_scalar.astype(mx.float32).item())
                 for layer in model.language_model.model.layers)


def read_attn_out_norm(model, layer: int):
    return model.language_model.model.layers[layer].post_attention_layernorm


def read_head_spec(model, layer: int, head: int) -> HeadSpec:
    attn = model.language_model.model.layers[layer].self_attn
    head_dim = int(attn.head_dim)
    n_heads = int(attn.n_heads)
    n_kv_heads = int(attn.n_kv_heads)
    kv_group = head * n_kv_heads // n_heads
    use_k_eq_v = bool(getattr(attn, "use_k_eq_v", False))
    W_Q_full = read_dense_weight(attn.q_proj)
    W_K_full = read_dense_weight(attn.k_proj)
    W_V_full = W_K_full if use_k_eq_v else read_dense_weight(attn.v_proj)
    W_O_full = read_dense_weight(attn.o_proj)
    return HeadSpec(
        layer=layer, head=head, kv_group=kv_group,
        head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        W_Q=W_Q_full[head * head_dim:(head + 1) * head_dim, :],
        W_K=W_K_full[kv_group * head_dim:(kv_group + 1) * head_dim, :],
        W_V=W_V_full[kv_group * head_dim:(kv_group + 1) * head_dim, :],
        W_O=W_O_full[:, head * head_dim:(head + 1) * head_dim],
        is_global=(attn.layer_type == "full_attention"),
        is_kv_shared=bool(attn.is_kv_shared_layer),
        use_k_eq_v=use_k_eq_v,
    )


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
    queries = dispatch(
        f"blocks.{layer_idx}.attn.q_pre_norm", layer_idx, "attn.q_pre_norm",
        queries, hooks, capture_set, cache,
    )
    queries = attn.q_norm(queries)

    if shared_kv is not None:
        keys, values = shared_kv
    else:
        offset = mx.array(c.offset) if c is not None else 0
        keys = attn.k_proj(x_normed).reshape(B, L, attn.n_kv_heads, attn.head_dim)
        keys = dispatch(
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
        keys = dispatch(
            f"blocks.{layer_idx}.attn.k_pre_rope", layer_idx, "attn.k_pre_rope",
            keys, hooks, capture_set, cache,
        )
        keys = attn.rope(keys, offset=offset)
        values = attn.v_norm(values)
        values = values.transpose(0, 2, 1, 3)
        if c is not None:
            keys, values = c.update_and_fetch(keys, values)

    queries = queries.transpose(0, 2, 1, 3)
    queries = dispatch(
        f"blocks.{layer_idx}.attn.q_pre_rope", layer_idx, "attn.q_pre_rope",
        queries, hooks, capture_set, cache,
    )
    queries = attn.rope(queries, offset=offset)

    queries = dispatch(
        f"blocks.{layer_idx}.attn.q", layer_idx, "attn.q", queries,
        hooks, capture_set, cache,
    )
    keys = dispatch(
        f"blocks.{layer_idx}.attn.k", layer_idx, "attn.k", keys,
        hooks, capture_set, cache,
    )
    values = dispatch(
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

    scores = apply_mask(scores, mask)

    scores = dispatch(
        f"blocks.{layer_idx}.attn.scores", layer_idx, "attn.scores", scores,
        hooks, capture_set, cache,
    )
    weights = mx.softmax(scores, axis=-1)
    weights = dispatch(
        f"blocks.{layer_idx}.attn.weights",
        layer_idx,
        "attn.weights",
        weights,
        hooks,
        capture_set,
        cache,
    )

    per_head_out = weights @ values_rep
    per_head_out = dispatch(
        f"blocks.{layer_idx}.attn.per_head_out",
        layer_idx,
        "attn.per_head_out",
        per_head_out,
        hooks,
        capture_set,
        cache,
    )

    output = per_head_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    output = dispatch(
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
    arch: Arch | None = None,
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
    h = dispatch("embed", None, "embed", h, hooks, capture_set, cache)
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
                "canonical forward. Only the dense checkpoints are wired."
            )
        c = kv_cache[i]
        local_mask = masks[i]
        shared_kv, offset = intermediates[previous_kvs[i]]
        per_layer_input = (
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        )

        h = dispatch(
            f"blocks.{i}.resid_pre", i, "resid_pre", h, hooks, capture_set, cache,
        )
        resid_pre = h

        x_normed = layer.input_layernorm(h)
        x_normed = dispatch(
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
        a = dispatch(
            f"blocks.{i}.attn_out", i, "attn_out", a, hooks, capture_set, cache,
        )
        h = resid_pre + a

        mid = h
        m = layer.pre_feedforward_layernorm(mid)
        m = dispatch(
            f"blocks.{i}.mlp.in_norm", i, "mlp.in_norm", m,
            hooks, capture_set, cache,
        )
        if i in manual_mlp_layer_set:
            gate = layer.mlp.gate_proj(m)
            gate = dispatch(
                f"blocks.{i}.mlp.gate", i, "mlp.gate", gate,
                hooks, capture_set, cache,
            )
            up = layer.mlp.up_proj(m)
            up = dispatch(
                f"blocks.{i}.mlp.up", i, "mlp.up", up,
                hooks, capture_set, cache,
            )
            act = nn.gelu_approx(gate)
            act = dispatch(
                f"blocks.{i}.mlp.act", i, "mlp.act", act,
                hooks, capture_set, cache,
            )
            down_in = act * up
            down_in = dispatch(
                f"blocks.{i}.mlp.down_in", i, "mlp.down_in", down_in,
                hooks, capture_set, cache,
            )
            m = layer.mlp.down_proj(down_in)
        else:
            m = layer.mlp(m)
        m = layer.post_feedforward_layernorm(m)
        m = dispatch(
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
            gate = dispatch(
                f"blocks.{i}.gate_out", i, "gate_out", gate,
                hooks, capture_set, cache,
            )
            h = h + gate

        if layer.layer_scalar is not None:
            h = h * layer.layer_scalar

        h = dispatch(
            f"blocks.{i}.resid_post", i, "resid_post", h, hooks, capture_set, cache,
        )

    u = read_unembed(model)
    logits = run_head(h, norm=tm.norm, unembed=lambda x: cap_logits(u, u.project(x)),
                      hooks=hooks, capture_set=capture_set, cache=cache)
    return logits, cache


ARCH = Architecture(
    model_type="gemma4", name="Gemma 4", loader="mlx-vlm",
    generate=True, score=True, train=True,
    layer_points=LAYER_HOOK_POINTS, global_points=GLOBAL_HOOK_POINTS,
    residual_law=("resid_post[i] == (resid_pre[i] + attn_out[i] + mlp_out[i] + gate_out[i])"
                  " * layer_scalar[i] == resid_pre[i+1]"),
    load=load_vlm,
    arch_of=read_arch,
    forward=run_forward,
    lm=read_language_model,
    prompt_cache=make_vlm_cache,
    head_logits=make_head_logits(read_unembed),
    project_to_logits=make_project_to_logits(read_unembed),
    tokenize=tokenize_vlm,
    attribution_unembed=read_unembed,
    head_weights=read_head_spec,
    dialect=DIALECT,
    reasoning=(Delimiters("<|channel>", "<channel|>", "thought\n"),),
    adapter_keys=ADAPTER_KEYS,
    refused_when=(Refusal(
        "enable_moe_block",
        "mixture-of-experts layers (Gemma 4 26B A4B) are not wired into the "
        "gemma4 forward; only the dense checkpoints load"),),
    absent_when=(Absence(
        "gate_out", "hidden_size_per_layer_input",
        "this checkpoint has no per-layer input embeddings (hidden_size_per_layer_input "
        "is 0, as in Gemma 4 31B), so no layer adds a per-layer gate"),),
    config_defaults={
        "hidden_size": 1536, "num_hidden_layers": 35, "num_attention_heads": 8,
        "num_key_value_heads": 1, "head_dim": 256, "vocab_size": 262144,
        "num_kv_shared_layers": 20, "hidden_size_per_layer_input": 256,
        "sliding_window_pattern": 5,
    },
    layer_scalars=read_layer_scalars,
    attn_out_norm=read_attn_out_norm,
)
