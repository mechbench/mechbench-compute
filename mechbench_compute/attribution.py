"""Direct Logit Attribution — decompose the residual stream into per-component
contributions and project each through the tied unembed to read off per-component
logit effects.

Canonical primitive for circuit analysis: "which heads (or layers, or branches)
contribute most to the model's prediction of token X vs. token Y at this
position?"

Four entry points:

  accumulated_resid(cache, layers=None, include_pre=False)
      Stack resid_post across layers: [n_layers, seq, d_model].

  decompose_resid(cache, layers=None)
      Per-branch contributions at each layer:
      {'attn': [L,S,D], 'mlp': [L,S,D], 'gate': [L,S,D]}.

  head_results(model, cache, layer)
      Per-head contribution to residual at one layer: [n_heads, seq, d_model].
      Computed by multiplying attn.per_head_out by the per-head slice of W_O.

  logit_attrs(model, residual_stack, target_token_ids, position=-1)
      Project each component in the stack through the tied unembed (linear,
      no final norm) and return per-component logit contributions to each
      target token at the given position.

Limitation vs. TransformerLens: this version projects through the *linear*
unembed only — no final-norm fold. TL optionally divides each component by
the final-norm's per-position scale factor before projecting, which makes
the decomposition exactly additive with the full-model output logits. Here
we skip that step. You get additive projections of the raw residual
components, which preserves the relative ranking of contributions (the
thing circuit-analysis actually uses) but does not sum to the model's
actual output logits. Adding norm-folding is a future extension — it
requires exposing the final norm's scale as a hook point.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import mlx.core as mx
import numpy as np

from .cache import ActivationCache
from .errors import CacheKeyError


def _layers_from_cache(
    cache: ActivationCache, *, point: str = "resid_post"
) -> list[int]:
    """Infer which layer indices the cache contains by counting
    `blocks.{i}.{point}` keys. The default probes `resid_post` because
    every layer-iterating helper assumes that's captured; pass a
    different point if the helper iterates a different one."""
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
    """If `layers` is given, use it. Otherwise infer from the cache —
    no module-level E4B-default fallback. (Previously defaulted to
    `range(N_LAYERS)` where N_LAYERS was hardcoded to 42.)"""
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
    """Stack a list of [B, S, D] tensors along a new leading axis and drop a
    batch dim of 1 if present. Returns float32 numpy."""
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
    """Stack `resid_post` across layers into `[n_layers, seq_len, d_model]`.

    If `include_pre=True`, layer 0's `resid_pre` is prepended, giving
    `[n_layers+1, seq_len, d_model]`. Useful for showing the embedding-layer
    starting point alongside each block's output.

    Requires `blocks.{i}.resid_post` captured for each `i` in `layers` (and
    `blocks.{layers[0]}.resid_pre` when `include_pre=True`).
    """
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
    """Per-branch contributions at each layer.

    Returns a dict with keys `'attn'`, `'mlp'`, `'gate'`, each mapping to an
    array of shape `[n_layers, seq_len, d_model]`. Requires
    `blocks.{i}.attn_out`, `blocks.{i}.mlp_out`, and `blocks.{i}.gate_out`
    captured for each layer.

    The MatFormer `gate_out` branch is specific to Gemma 4's per-layer-input
    side-channel. It is load-bearing at global-attention layers (see
    finding 03) and should not be omitted from decomposition.
    """
    layer_list = _resolve_layers(layers, cache, point="attn_out")
    out: dict[str, np.ndarray] = {}
    for branch in ("attn_out", "mlp_out", "gate_out"):
        tensors = [_require(cache, f"blocks.{i}.{branch}") for i in layer_list]
        out[branch.removesuffix("_out")] = _stack_and_squeeze(tensors)
    return out


def head_results(model, cache: ActivationCache, layer: int) -> np.ndarray:
    """Per-head contribution to the residual stream at one layer.

    Returns `[n_heads, seq_len, d_model]` (float32). Computed by multiplying
    each head's `per_head_out` (the pre-concat attention output) by the
    corresponding column-block of `W_O`. The head outputs sum to the layer's
    attention contribution *before* the post-attention layernorm applied by
    the layer; this function gives you the pre-norm per-head writes so you
    can ask "which head writes the most to the (paris - berlin) direction?"

    Requires `blocks.{layer}.attn.per_head_out` captured.
    """
    per_head = _require(cache, f"blocks.{layer}.attn.per_head_out").astype(
        mx.float32
    )
    mx.eval(per_head)
    per_head_np = np.array(per_head)
    if per_head_np.ndim == 4 and per_head_np.shape[0] == 1:
        per_head_np = per_head_np[0]  # [n_heads, L, head_dim]

    block = model._model.language_model.model.layers[layer]
    head_dim = int(block.self_attn.head_dim)
    # Dequantize o_proj if the model is quantized (the only published 12B
    # conversions are 8-bit, whose .weight is uint32-packed); otherwise the
    # column-slicing below would index into packed data and corrupt the result.
    o_proj = block.self_attn.o_proj
    o_weight = o_proj.weight
    if hasattr(o_proj, "scales"):
        o_weight = mx.dequantize(
            o_weight, o_proj.scales, o_proj.biases,
            group_size=o_proj.group_size, bits=o_proj.bits,
        )
    W_O_full = np.array(o_weight.astype(mx.float32))
    # W_O_full shape: [d_model, n_heads * head_dim]
    n_heads = per_head_np.shape[0]
    seq_len = per_head_np.shape[1]
    d_model = W_O_full.shape[0]

    out = np.zeros((n_heads, seq_len, d_model), dtype=np.float32)
    for h in range(n_heads):
        W_O_h = W_O_full[:, h * head_dim : (h + 1) * head_dim]  # [d_model, head_dim]
        out[h] = per_head_np[h] @ W_O_h.T
    return out


def _final_norm_gain(model) -> np.ndarray:
    """The final norm's learned gain as a vector — (1 + w) for the
    Gemma families, w for llama/qwen. Elementwise-linear, so it can be
    folded into each component before the unembed."""
    if model.arch.model_type in ("qwen2", "llama"):
        w = model._model.model.norm.weight
        offset = 0.0
    else:
        w = model._model.language_model.model.norm.weight
        offset = 1.0
    arr = np.array(mx.array(w).astype(mx.float32))
    return arr + offset


def logit_attrs(
    model,
    residual_stack: np.ndarray,
    target_token_ids: Sequence[int],
    *,
    position: int = -1,
    apply_ln: bool = False,
    ln_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Project each component in `residual_stack` through the tied unembed
    (linear, no final norm) and return logit contributions to each target
    token at the given sequence position.

    Args:
        model: The `Model` instance (used only for its unembed weights).
        residual_stack: Any array shaped `[..., seq_len, d_model]` — for
            example the `[n_layers, S, D]` returned by `accumulated_resid`,
            the `[n_heads, S, D]` returned by `head_results`, or a single
            `[S, D]` branch contribution. Leading dims are preserved.
        target_token_ids: Tokens to attribute. The most common pattern is
            a pair of competing tokens (e.g. `[paris_id, berlin_id]`) and
            then reading `attrs[..., 0] - attrs[..., 1]` as the
            contribution to the logit-difference.
        position: Sequence position to read at. Default `-1` (final token).
        apply_ln: Fold the final RMSNorm (captured scale + learned gain)
            into each component, making the decomposition sum to the
            model's true final logits (task 000142).
        ln_scale: The captured `final_norm.scale` value ([B, S] or [S])
            from the same run. Required when apply_ln=True.

    Returns:
        `np.ndarray` of shape `[..., len(target_token_ids)]`, float32.
    """
    stack_at_pos = residual_stack[..., position, :]  # [..., d_model]
    leading_shape = stack_at_pos.shape[:-1]
    d_model = stack_at_pos.shape[-1]
    flat = stack_at_pos.reshape(-1, d_model)  # [N, d_model]
    if apply_ln:
        # TransformerLens's apply_ln semantics (task 000142): divide by
        # the CAPTURED per-position rms and fold in the norm's gain —
        # both elementwise-linear, so summing the per-component results
        # reproduces the model's true final logit.
        if ln_scale is None:
            raise ValueError(
                "apply_ln=True needs ln_scale — capture "
                "'final_norm.scale' (Capture.final_norm_scale()) in the "
                "same run and pass its value here"
            )
        scale = float(np.asarray(ln_scale, dtype=np.float64).reshape(-1)[position])
        flat = (flat / scale) * _final_norm_gain(model)
    v = mx.array(flat, dtype=mx.bfloat16)

    # Project through the unembed without applying the final norm — DLA
    # reads per-component contributions, and the final norm would mix
    # components nonlinearly. Per-family dispatch on where the unembed
    # weights live (mlx-vlm Gemma vs mlx-lm Qwen).
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
        else:  # gemma4: tied embed.as_linear
            logits = tm.embed_tokens.as_linear(v)

    logits = logits.astype(mx.float32)
    mx.eval(logits)
    logits_np = np.array(logits)  # [N, vocab]

    target_ids = np.asarray(target_token_ids, dtype=np.int64)
    attrs = logits_np[:, target_ids]  # [N, T]
    return attrs.reshape(*leading_shape, len(target_ids))
