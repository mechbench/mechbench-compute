from __future__ import annotations

import mlx.core as mx


def apply_mask(scores: mx.array, mask) -> mx.array:
    if mask is None:
        return scores
    q_len, k_len = scores.shape[-2], scores.shape[-1]
    if isinstance(mask, str):
        if mask != "causal":
            raise NotImplementedError(
                f"Unknown mask type in manual attention path: {mask!r}. "
                f"Expected None, 'causal', or an mx.array."
            )
        i = mx.arange(q_len).reshape(q_len, 1)
        j = mx.arange(k_len).reshape(1, k_len)
        mask = j <= (k_len - q_len + i)
    if not isinstance(mask, mx.array):
        raise NotImplementedError(
            f"Unknown mask type in manual attention path: {type(mask).__name__}. "
            f"Expected None, 'causal', or an mx.array."
        )
    if mask.shape[-1] != k_len:
        mask = mask[..., -k_len:]
    if mask.dtype == mx.bool_:
        mask = mx.where(mask, mx.array(0.0, dtype=scores.dtype),
                        mx.array(-1e9, dtype=scores.dtype))
    return scores + mask.astype(scores.dtype)
