from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.dictionaries.train_against import Against
from mechbench_compute.dictionaries.train_fidelity import (
    encode,
    next_token_loss,
    rounded,
)
from mechbench_compute.dictionaries.train_harvest import read_kept
from mechbench_compute.dictionaries.train_settings import DENSE_RATE, Settings
from mechbench_compute.protocol.serialize_model import serialize_model
from mechbench_compute.tensors import ShardWriter, collection


def read_scales(x: Any, models: int) -> np.ndarray:
    d = int(x.shape[1]) // models
    return np.asarray([np.sqrt(d / float(mx.mean(mx.sum(x[:, m * d:(m + 1) * d] ** 2, axis=-1)).item()))
                       for m in range(models)], dtype=np.float64)


def spread_scales(scales: np.ndarray, width: int) -> np.ndarray:
    return np.repeat(np.asarray(scales, dtype=np.float64), width // len(scales))


def replace_kept(recon: Any, mask: Any):
    def hook(act, info):
        return mx.where(mask, recon[None].astype(act.dtype), act)

    return hook


def zero_kept(mask: Any):
    def hook(act, info):
        return mx.where(mask, mx.zeros_like(act), act)

    return hook


def read_side(model: Any, name: str, ids: np.ndarray) -> tuple[Any, Any]:
    out = model.run(mx.array(ids), capture=[name])
    return out.cache[name][0].astype(mx.float32), out.logits


def score_side(model: Any, name: str, ids: np.ndarray, recon: Any, mask: Any,
               on_unit: Callable[[], None] | None) -> tuple[float, float]:
    losses = []
    for hook in (replace_kept(recon, mask), zero_kept(mask)):
        out = model.run(mx.array(ids), hooks={name: hook})
        losses.append(next_token_loss(out.logits, ids[0])[0])
        if on_unit is not None:
            on_unit()
    return losses[0], losses[1]


def measure_crosscoder(model: Any, name: str, windows: Sequence[Sequence[int]],
                       folded: Mapping[str, np.ndarray], bos: int | None, against: Against,
                       on_unit: Callable[[], None] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    weights = {k: mx.array(v) for k, v in folded.items()}
    width = int(folded["w_enc"].shape[1])
    d = int(folded["w_dec"].shape[1]) // 2
    fired = np.zeros(width, dtype=np.int64)
    summed = np.zeros(width, dtype=np.float64)
    sse, total = np.zeros(2), np.zeros(2)
    sums = np.zeros((2, d))
    losses = np.zeros((2, 3))
    active = 0.0
    n = predicted = 0
    for window in windows:
        ids = np.asarray([window], dtype=np.int32)
        kept = read_kept(ids, bos)
        mask_np = np.zeros(ids.shape[1], dtype=bool)
        mask_np[kept] = True
        mask = mx.array(mask_np)[None, :, None]
        acts, logits = [None, None], [None, None]
        acts[0], logits[0] = read_side(model, name, ids)
        if on_unit is not None:
            on_unit()
        with against.swap():
            acts[1], logits[1] = read_side(model, name, ids)
            if on_unit is not None:
                on_unit()
            x = mx.concatenate(acts, axis=-1)
            f = encode(x, weights)
            recon = f @ weights["w_dec"] + weights["b_dec"]
            parts = (recon[:, :d], recon[:, d:])
            against_scores = score_side(model, name, ids, parts[1], mask, on_unit)
        model_scores = score_side(model, name, ids, parts[0], mask, on_unit)
        rows = mx.array(kept)
        for m in range(2):
            a, r = acts[m][rows], parts[m][rows]
            sse[m] += float(mx.sum((a - r) ** 2).item())
            total[m] += float(mx.sum(a ** 2).item())
            sums[m] += np.array(mx.sum(a, axis=0), dtype=np.float64)
            clean, count = next_token_loss(logits[m], ids[0])
            losses[m] += (clean, *(model_scores if m == 0 else against_scores))
        predicted += count
        fk = f[rows]
        nonzero = fk != 0
        active += float(mx.sum(nonzero).item())
        fired += np.array(mx.sum(nonzero, axis=0), dtype=np.int64)
        summed += np.array(mx.sum(fk, axis=0), dtype=np.float64)
        n += int(kept.size)
    spread = total - np.sum(sums ** 2, axis=1) / n if n else np.zeros(2)
    joint = float(spread.sum())
    rate = fired / n if n else np.zeros(width)
    fidelity = {
        "tokens": n,
        "variance_explained": round(1.0 - float(sse.sum()) / joint, 6) if joint > 0 else None,
        "mse": round(float(sse.sum()) / (n * 2 * d), 9) if n else None,
        "l0": round(active / n, 6) if n else None,
        "dead": round(float(np.mean(fired == 0)), 6),
        "dense": round(float(np.mean(rate > DENSE_RATE)), 6),
        "dense_rate": DENSE_RATE,
        "reads": [read_model_fidelity(sse[m], spread[m], n, d, losses[m], predicted) for m in range(2)],
    }
    return fidelity, {"fired": fired, "summed": summed, "tokens": np.asarray(n)}


def read_model_fidelity(sse: float, spread: float, n: int, d: int, losses: np.ndarray,
                        predicted: int) -> dict[str, Any]:
    clean, spliced, zero = (losses / predicted).tolist() if predicted else (None, None, None)
    delta = zero_delta = recovered = None
    if predicted:
        delta, zero_delta = spliced - clean, zero - clean
        recovered = 1.0 - delta / zero_delta if zero_delta else None
    return {
        "variance_explained": round(1.0 - float(sse) / float(spread), 6) if spread > 0 else None,
        "mse": round(float(sse) / (n * d), 9) if n else None,
        "loss": {"clean": rounded(clean), "spliced": rounded(spliced), "zero": rounded(zero),
                 "delta": rounded(delta), "zero_delta": rounded(zero_delta),
                 "recovered": rounded(recovered), "predicted": predicted},
    }


def read_decoder(trainer: Any) -> np.ndarray:
    return np.array(trainer.params["w_dec"])


def read_difference(w_dec: np.ndarray) -> np.ndarray:
    d = w_dec.shape[1] // 2
    w = np.asarray(w_dec, dtype=np.float64)
    mine, theirs = np.linalg.norm(w[:, :d], axis=1), np.linalg.norm(w[:, d:], axis=1)
    both = mine + theirs
    return (mine - theirs) / np.where(both > 0, both, 1.0)


def emit_crosscoder(model: Any, s: Settings, folded: Mapping[str, np.ndarray], trained_w_dec: np.ndarray,
                    ref: Any, against: Against, stream: Any, fidelity: Mapping[str, Any],
                    readout: Mapping[str, np.ndarray], *, model_id: str | None,
                    trained: dict[str, Any]) -> dict[str, Any]:
    width = int(folded["w_enc"].shape[1])
    d = int(folded["w_dec"].shape[1]) // 2
    reads = [{"model": model_id, "adapters": adapters, "layer": s.layer, "point": s.point, "head": None, "d": d}
             for adapters in (against.model_adapters, against.adapters)]
    writer = ShardWriter(tempfile.mkdtemp(prefix="mechbench-tensor-"))
    encoders = np.ascontiguousarray(folded["w_enc"].T)
    w_dec = folded["w_dec"]
    norms = np.linalg.norm(w_dec.astype(np.float64)[:, :d], axis=1)
    against_norms = np.linalg.norm(w_dec.astype(np.float64)[:, d:], axis=1)
    difference = read_difference(trained_w_dec)
    tokens = int(readout["tokens"])
    for i in range(width):
        fired = int(readout["fired"][i])
        item = {"id": str(i), "index": i, "vector": w_dec[i, :d], "encoder": encoders[i, :d],
                "norm": float(norms[i]), "b_enc": float(folded["b_enc"][i]),
                "threshold": float(folded["threshold"][i]),
                "against_vector": np.ascontiguousarray(w_dec[i, d:]),
                "against_encoder": np.ascontiguousarray(encoders[i, d:]),
                "against_norm": float(against_norms[i]), "difference": round(float(difference[i]), 6),
                "frequency": round(fired / tokens, 9) if tokens else 0.0}
        if fired:
            item["mean_activation"] = round(float(readout["summed"][i]) / fired, 6)
        writer.add(item)
    activation: dict[str, Any] = {"fn": "jumprelu", "training": s.activation}
    activation.update({"k": s.k} if s.activation == "batchtopk" else {"l0": s.l0})
    b_dec = folded["b_dec"].astype(np.float64)
    return collection(
        "direction/dictionary", writer.close(),
        derivation="crosscoder",
        reads=reads, writes=[dict(r) for r in reads],
        model={"id": model_id, "architecture": getattr(getattr(model, "arch", None), "model_type", None)},
        width=width, d_in=d, d_out=d,
        activation=activation,
        b_dec=[float(x) for x in b_dec[:d]],
        against_b_dec=[float(x) for x in b_dec[d:]],
        trained={"corpus": stream.describe(), "model": serialize_model(ref, model), "against": against.wire,
                 "hyperparameters": s.hyperparameters(), **trained},
        fidelity=fidelity,
    )
