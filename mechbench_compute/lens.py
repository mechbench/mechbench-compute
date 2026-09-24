from __future__ import annotations

from typing import Iterable, Optional

import mlx.core as mx
import numpy as np

from .attribution import _layers_from_cache


def _resolve_layers(
    layers: Optional[Iterable[int]], cache=None
) -> list[int]:
    if layers is not None:
        return list(layers)
    if cache is None:
        raise ValueError(
            "_resolve_layers needs `layers` or a `cache` to infer from"
        )
    return _layers_from_cache(cache, point="resid_post")


def logit_lens_final(
    model,
    cache,
    target_id: int,
    *,
    layers: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    layers_list = _resolve_layers(layers, cache)
    n = len(layers_list)
    ranks = np.zeros(n, dtype=np.int64)
    logprobs = np.zeros(n, dtype=np.float64)

    for k, i in enumerate(layers_list):
        resid = cache[f"blocks.{i}.resid_post"]
        logits_i = model.project_to_logits(resid)
        last = logits_i[0, -1, :].astype(mx.float32)
        lp = last - mx.logsumexp(last)
        mx.eval(lp)
        lp_np = np.array(lp)
        target_lp = float(lp_np[target_id])
        ranks[k] = int(np.sum(lp_np > target_lp))
        logprobs[k] = target_lp

    return ranks, logprobs


def logit_lens_per_position(
    model,
    cache,
    target_id: int,
    *,
    layers: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    layers_list = _resolve_layers(layers, cache)
    first = cache[f"blocks.{layers_list[0]}.resid_post"]
    seq_len = first.shape[1]
    n = len(layers_list)

    ranks = np.zeros((n, seq_len), dtype=np.int64)
    logprobs = np.zeros((n, seq_len), dtype=np.float64)

    for k, i in enumerate(layers_list):
        resid = cache[f"blocks.{i}.resid_post"]
        logits_i = model.project_to_logits(resid)
        f32 = logits_i[0].astype(mx.float32)
        lp = f32 - mx.logsumexp(f32, axis=-1, keepdims=True)
        mx.eval(lp)
        lp_np = np.array(lp)
        for pos in range(seq_len):
            target_lp = float(lp_np[pos, target_id])
            ranks[k, pos] = int(np.sum(lp_np[pos] > target_lp))
            logprobs[k, pos] = target_lp

    return ranks, logprobs
