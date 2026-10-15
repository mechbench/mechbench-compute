from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.dictionaries.train_harvest import read_kept
from mechbench_compute.dictionaries.train_settings import DENSE_RATE


def encode(x: Any, weights: Mapping[str, Any]) -> Any:
    pre = x @ weights["w_enc"] + weights["b_enc"]
    return pre * (pre > weights["threshold"])


def next_token_loss(logits: Any, ids: np.ndarray) -> tuple[float, int]:
    logits = logits[0, 1:-1].astype(mx.float32)
    target = mx.array(ids[2:].astype(np.int32))
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logp, target[:, None], axis=-1)
    return float(-mx.sum(picked).item()), int(target.shape[0])


def splice(weights: Mapping[str, Any], mask: Any, mode: str):
    def hook(act, info):
        x = act.astype(mx.float32)
        new = mx.zeros_like(x) if mode == "zero" else encode(x, weights) @ weights["w_dec"] + weights["b_dec"]
        return mx.where(mask, new, x).astype(act.dtype)

    return hook


def measure(model: Any, name: str, windows: Sequence[Sequence[int]], folded: Mapping[str, np.ndarray],
            bos: int | None, on_unit: Callable[[], None] | None = None) -> dict[str, Any]:
    weights = {k: mx.array(v) for k, v in folded.items()}
    width = int(folded["w_enc"].shape[1])
    fired = np.zeros(width, dtype=np.int64)
    sse = total = active = 0.0
    sums: Any = None
    n = 0
    losses = {"clean": 0.0, "spliced": 0.0, "zero": 0.0}
    predicted = 0
    for window in windows:
        ids = np.asarray([window], dtype=np.int32)
        kept = read_kept(ids, bos)
        mask_np = np.zeros(ids.shape[1], dtype=bool)
        mask_np[kept] = True
        mask = mx.array(mask_np)[None, :, None]
        clean = model.run(mx.array(ids), capture=[name])
        acts = clean.cache[name].reshape(-1, clean.cache[name].shape[-1])[mx.array(kept)].astype(mx.float32)
        f = encode(acts, weights)
        recon = f @ weights["w_dec"] + weights["b_dec"]
        sse += float(mx.sum((acts - recon) ** 2).item())
        total += float(mx.sum(acts ** 2).item())
        row = np.array(mx.sum(acts, axis=0), dtype=np.float64)
        sums = row if sums is None else sums + row
        nonzero = f != 0
        active += float(mx.sum(nonzero).item())
        fired += np.array(mx.sum(nonzero, axis=0), dtype=np.int64)
        n += int(acts.shape[0])
        value, count = next_token_loss(clean.logits, ids[0])
        losses["clean"] += value
        predicted += count
        if on_unit is not None:
            on_unit()
        for mode in ("spliced", "zero"):
            out = model.run(mx.array(ids), hooks={name: splice(weights, mask, mode)})
            losses[mode] += next_token_loss(out.logits, ids[0])[0]
            if on_unit is not None:
                on_unit()
    spread = total - float(np.sum(sums ** 2)) / n if n else 0.0
    mean = {k: v / predicted for k, v in losses.items()} if predicted else dict.fromkeys(losses)
    delta = zero_delta = recovered = None
    if predicted:
        delta = mean["spliced"] - mean["clean"]
        zero_delta = mean["zero"] - mean["clean"]
        recovered = 1.0 - delta / zero_delta if zero_delta else None
    rate = fired / n if n else np.zeros(width)
    return {
        "tokens": n,
        "variance_explained": round(1.0 - sse / spread, 6) if spread > 0 else None,
        "mse": round(sse / (n * acts.shape[1]), 9) if n else None,
        "l0": round(active / n, 6) if n else None,
        "dead": round(float(np.mean(fired == 0)), 6),
        "dense": round(float(np.mean(rate > DENSE_RATE)), 6),
        "dense_rate": DENSE_RATE,
        "loss": {"clean": rounded(mean["clean"]), "spliced": rounded(mean["spliced"]),
                 "zero": rounded(mean["zero"]), "delta": rounded(delta), "zero_delta": rounded(zero_delta),
                 "recovered": rounded(recovered), "predicted": predicted},
    }


def rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)
