from __future__ import annotations

from typing import Iterable, Optional, Sequence

import mlx.core as mx
import numpy as np

from .cache import ActivationCache
from .errors import CacheKeyError


def _layers_from_cache(
    cache: ActivationCache, *, point: str = "resid_post"
) -> list[int]:
    indices: list[int] = []
    prefix = "blocks."
    suffix = f".{point}"
    for k in cache.keys():
        if k.startswith(prefix) and k.endswith(suffix):
            try:
                i = int(k[len(prefix) : -len(suffix)])
            except ValueError:
                continue
            indices.append(i)
    if not indices:
        raise CacheKeyError(f"blocks.*.{point}", cache.keys())
    return sorted(set(indices))


def _resolve_layers(
    layers: Optional[Iterable[int]],
    cache: Optional[ActivationCache] = None,
    *,
    point: str = "resid_post",
) -> list[int]:
    if layers is not None:
        return list(layers)
    if cache is None:
        raise ValueError(
            "_resolve_layers needs `layers` or a `cache` to infer from"
        )
    return _layers_from_cache(cache, point=point)


def _require(cache: ActivationCache, key: str) -> mx.array:
    if key not in cache:
        raise CacheKeyError(key, cache.keys())
    return cache[key]


def _stack_and_squeeze(tensors: list[mx.array]) -> np.ndarray:
    arr = mx.stack([t.astype(mx.float32) for t in tensors], axis=0)
    mx.eval(arr)
    out = np.array(arr)
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    return out


def accumulated_resid(
    cache: ActivationCache,
    *,
    layers: Optional[Iterable[int]] = None,
    include_pre: bool = False,
) -> np.ndarray:
    layer_list = _resolve_layers(layers, cache, point="resid_post")
    tensors: list[mx.array] = []
    if include_pre:
        tensors.append(_require(cache, f"blocks.{layer_list[0]}.resid_pre"))
    for i in layer_list:
        tensors.append(_require(cache, f"blocks.{i}.resid_post"))
    return _stack_and_squeeze(tensors)


def decompose_resid(
    cache: ActivationCache,
    *,
    layers: Optional[Iterable[int]] = None,
) -> dict[str, np.ndarray]:
    layer_list = _resolve_layers(layers, cache, point="attn_out")
    out: dict[str, np.ndarray] = {}
    for branch in ("attn_out", "mlp_out", "gate_out"):
        tensors = [_require(cache, f"blocks.{i}.{branch}") for i in layer_list]
        out[branch.removesuffix("_out")] = _stack_and_squeeze(tensors)
    return out


def head_results(model, cache: ActivationCache, layer: int) -> np.ndarray:
    per_head = _require(cache, f"blocks.{layer}.attn.per_head_out").astype(
        mx.float32
    )
    mx.eval(per_head)
    per_head_np = np.array(per_head)
    if per_head_np.ndim == 4 and per_head_np.shape[0] == 1:
        per_head_np = per_head_np[0]

    block = model._model.language_model.model.layers[layer]
    head_dim = int(block.self_attn.head_dim)
    o_proj = block.self_attn.o_proj
    o_weight = o_proj.weight
    if hasattr(o_proj, "scales"):
        o_weight = mx.dequantize(
            o_weight, o_proj.scales, o_proj.biases,
            group_size=o_proj.group_size, bits=o_proj.bits,
        )
    W_O_full = np.array(o_weight.astype(mx.float32))
    n_heads = per_head_np.shape[0]
    seq_len = per_head_np.shape[1]
    d_model = W_O_full.shape[0]

    out = np.zeros((n_heads, seq_len, d_model), dtype=np.float32)
    for h in range(n_heads):
        W_O_h = W_O_full[:, h * head_dim : (h + 1) * head_dim]
        out[h] = per_head_np[h] @ W_O_h.T
    return out


def _read_gain_offset(norm) -> float:
    from mlx import nn
    from mlx_vlm.models.gemma3.language import RMSNorm as Gemma3RMSNorm

    offsets = {nn.RMSNorm: 0.0, Gemma3RMSNorm: 1.0}
    if type(norm) not in offsets:
        raise NotImplementedError(
            f"apply_ln does not know the gain of a final norm of type "
            f"{type(norm).__module__}.{type(norm).__qualname__}; known: "
            + ", ".join(f"{t.__module__}.{t.__qualname__}" for t in offsets))
    return offsets[type(norm)]


def _final_norm_gain(model) -> np.ndarray:
    if model.arch.model_type in ("qwen2", "llama"):
        norm = model._model.model.norm
    else:
        norm = model._model.language_model.model.norm
    arr = np.array(mx.array(norm.weight).astype(mx.float32))
    return arr + _read_gain_offset(norm)


def logit_attrs(
    model,
    residual_stack: np.ndarray,
    target_token_ids: Sequence[int],
    *,
    position: int = -1,
    apply_ln: bool = False,
    ln_scale: np.ndarray | None = None,
) -> np.ndarray:
    stack_at_pos = residual_stack[..., position, :]
    leading_shape = stack_at_pos.shape[:-1]
    d_model = stack_at_pos.shape[-1]
    flat = stack_at_pos.reshape(-1, d_model)
    if apply_ln:
        if ln_scale is None:
            raise ValueError(
                "apply_ln=True needs ln_scale — capture "
                "'final_norm.scale' (Capture.final_norm_scale()) in the "
                "same run and pass its value here"
            )
        scale = float(np.asarray(ln_scale, dtype=np.float64).reshape(-1)[position])
        flat = (flat / scale) * _final_norm_gain(model)
    v = mx.array(flat, dtype=mx.float32)

    if model.arch.model_type in ("qwen2", "llama"):
        if model._model.args.tie_word_embeddings:
            logits = model._model.model.embed_tokens.as_linear(v)
        else:
            logits = model._model.lm_head(v)
    else:
        lm = model._model.language_model
        tm = lm.model
        if model.arch.model_type == "gemma3":
            logits = lm.lm_head(v)
        else:
            logits = tm.embed_tokens.as_linear(v)

    logits = logits.astype(mx.float32)
    mx.eval(logits)
    logits_np = np.array(logits)

    target_ids = np.asarray(target_token_ids, dtype=np.int64)
    attrs = logits_np[:, target_ids]
    return attrs.reshape(*leading_shape, len(target_ids))
