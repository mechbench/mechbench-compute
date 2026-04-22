"""Logit lens helpers.

Project the residual stream at each layer through the tied unembed and read
off what the model 'thinks' at that depth. Two flavors:

  logit_lens_final         - per-layer (rank, logprob) at the final position
                             (the standard logit lens, used by step_01)
  logit_lens_per_position  - same but at every position, returns [n_layers x seq_len]
                             (used by step_08)

Both consume an ActivationCache populated by Capture.residual(layers, point='post').
"""

from __future__ import annotations

from typing import Iterable, Optional

import mlx.core as mx
import numpy as np

from ._arch import N_LAYERS


def _resolve_layers(layers: Optional[Iterable[int]]) -> list[int]:
    return list(range(N_LAYERS)) if layers is None else list(layers)


def logit_lens_final(
    model,
    cache,
    target_id: int,
    *,
    layers: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project resid_post at each requested layer through the unembed; return
    (ranks, logprobs) of target_id at the final sequence position.

    Args:
        model: Model instance (for project_to_logits).
        cache: ActivationCache containing blocks.{i}.resid_post for each
            layer in `layers`. Typically produced by Model.run with
            interventions=[Capture.residual(layers=range(N_LAYERS))].
        target_id: The token id whose trajectory to follow.
        layers: Layers to project, in order. Default: 0..N_LAYERS-1.

    Returns:
        (ranks, logprobs) — both np.ndarray of shape [len(layers)].
        ranks[k] is the rank of target_id at layers[k] (0 = top-1).
        logprobs[k] is the normalized log-probability of target_id.
    """
    layers_list = _resolve_layers(layers)
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
    """Project resid_post at each requested layer through the unembed at every
    sequence position. Returns matrices of shape [len(layers), seq_len].

    Used by step_08 to ask 'where in the sequence and at what depth does
    the answer become visible?'

    Args:
        model: Model instance.
        cache: ActivationCache containing blocks.{i}.resid_post for each
            layer in `layers`.
        target_id: The token id whose trajectory to follow.
        layers: Layers to project. Default: 0..N_LAYERS-1.

    Returns:
        (ranks, logprobs) — np.ndarray of shape [len(layers), seq_len].
    """
    layers_list = _resolve_layers(layers)
    # Read seq_len from the first layer's cache entry.
    first = cache[f"blocks.{layers_list[0]}.resid_post"]
    seq_len = first.shape[1]
    n = len(layers_list)

    ranks = np.zeros((n, seq_len), dtype=np.int64)
    logprobs = np.zeros((n, seq_len), dtype=np.float64)

    for k, i in enumerate(layers_list):
        resid = cache[f"blocks.{i}.resid_post"]
        logits_i = model.project_to_logits(resid)
        f32 = logits_i[0].astype(mx.float32)  # [seq_len, vocab]
        lp = f32 - mx.logsumexp(f32, axis=-1, keepdims=True)
        mx.eval(lp)
        lp_np = np.array(lp)
        for pos in range(seq_len):
            target_lp = float(lp_np[pos, target_id])
            ranks[k, pos] = int(np.sum(lp_np[pos] > target_lp))
            logprobs[k, pos] = target_lp

    return ranks, logprobs
