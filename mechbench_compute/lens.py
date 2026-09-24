from __future__ import annotations

from typing import Iterable, Optional

import mlx.core as mx
import numpy as np

from .attribution import _layers_from_cache
from .interp.answer import Answer, make_answer


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
    target: int | Answer,
    *,
    layers: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    answer = _read_answer(target)
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
        ranks[k] = answer.rank(lp_np)
        logprobs[k] = answer.logp(lp_np)

    return ranks, logprobs


def logit_lens_per_position(
    model,
    cache,
    target: int | Answer,
    *,
    layers: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    answer = _read_answer(target)
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
            ranks[k, pos] = answer.rank(lp_np[pos])
            logprobs[k, pos] = answer.logp(lp_np[pos])

    return ranks, logprobs


def _read_answer(target: int | Answer) -> Answer:
    return target if isinstance(target, Answer) else make_answer([int(target)])
