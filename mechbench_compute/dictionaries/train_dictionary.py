from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.dictionaries.train_corpus import open_stream
from mechbench_compute.dictionaries.train_fidelity import measure
from mechbench_compute.dictionaries.train_harvest import fill, read_point_name
from mechbench_compute.dictionaries.train_sae import Trainer, fold, init_params
from mechbench_compute.dictionaries.train_settings import Settings, read_settings
from mechbench_compute.dictionaries.train_state import (
    CURVE_FIELDS,
    capture_state,
    restore_state,
)
from mechbench_compute.protocol.serialize_model import serialize_model
from mechbench_compute.tensors import ShardWriter, collection


def read_model_id(model: Any) -> str | None:
    for attr in ("requested_ref", "requested", "model_id"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    v = getattr(getattr(model, "arch", None), "model_id", None)
    return v if isinstance(v, str) and v else None


def hash_windows(windows: Sequence[Sequence[int]]) -> bytes:
    return hashlib.sha256(np.asarray(windows, dtype="<i4").tobytes()).digest()


def train_dictionary(model: Any, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any], *,
                     on_start: Callable[..., None] | None = None,
                     on_item: Callable[..., None] | None = None,
                     on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
                     resume_state: Mapping[str, Any] | None = None,
                     on_step: Callable[[dict[str, float]], None] | None = None) -> dict[str, Any]:
    from mechbench_compute import __version__

    s = read_settings(params)
    name = read_point_name(model, s.point, s.layer)
    tokenizer = model.tokenizer
    bos = getattr(tokenizer, "bos_token_id", None)
    stream = open_stream(s.corpus, records, tokenizer, seed=s.seed, context=s.context)
    rng = np.random.default_rng([s.seed, 1])
    total = 2 * s.steps + 3 * s.heldout_windows
    if on_start:
        on_start(total)
    trainer: Trainer | None = None
    if resume_state:
        weights = resume_state["weights"]
        trainer = Trainer(s, {k: mx.array(np.asarray(v, dtype=np.float32)) for k, v in weights.items()},
                          int(np.asarray(weights["w_dec"]).shape[1]))
        restored = restore_state(trainer, stream, rng, resume_state)
        scale, tokens, curve = restored["scale"], restored["tokens"], restored["curve"]
        heldout, first_chunk = restored["heldout"], restored["first_chunk"]
        if on_item:
            for _ in range(2 * trainer.step):
                on_item(None, None, True)
    else:
        heldout = np.asarray([stream.next_window() for _ in range(s.heldout_windows)], dtype=np.int64)
        scale, tokens, curve, first_chunk = 0.0, 0, [], b""

    def unit() -> None:
        if on_item:
            on_item()

    while trainer is None or trainer.step < s.steps:
        n_steps = min(s.steps_per_chunk, s.steps - (trainer.step if trainer else 0))
        seen: list[list[int]] = []
        x = fill(model, name, stream, needed=n_steps * s.batch_tokens, rows=s.rows, bos=bos,
                 unit=s.batch_tokens, on_unit=unit, on_windows=seen.extend if trainer is None else None)
        if trainer is None:
            first_chunk = hash_windows(seen)
            scale = math.sqrt(x.shape[1] / float(mx.mean(mx.sum(x * x, axis=-1)).item()))
            x = x * scale
            trainer = Trainer(s, init_params(s, x, s.seed), int(x.shape[1]))
        else:
            x = x * scale
        x = x[mx.array(rng.permutation(int(x.shape[0])).astype(np.int32))]
        mx.eval(x)
        for j in range(n_steps):
            metrics = trainer.train(x[j * s.batch_tokens:(j + 1) * s.batch_tokens])
            tokens += s.batch_tokens
            metrics["tokens"] = tokens
            if trainer.step % s.curve_every == 0 or trainer.step == s.steps:
                curve.append({f: metrics[f] for f in CURVE_FIELDS})
            if on_step:
                on_step(metrics)
            unit()
        chunk = n_steps * s.batch_tokens
        if (on_checkpoint is not None and trainer.step < s.steps
                and tokens // s.checkpoint_tokens > (tokens - chunk) // s.checkpoint_tokens):
            on_checkpoint(capture_state(trainer, stream, rng, scale=scale, tokens=tokens, heldout=heldout,
                                        first_chunk=first_chunk, curve=curve))
    folded = fold(trainer.params, trainer.thresholds(), scale)
    fidelity = measure(model, name, heldout.tolist(), folded, bos, on_unit=unit)
    return emit(model, s, folded, params.get("model"), stream, fidelity, trained={
        "tokens": tokens, "steps": trainer.step, "scale": scale, "seed": s.seed,
        "first_chunk_sha256": first_chunk.hex(), "heldout_sha256": hash_windows(heldout).hex(),
        "heldout_windows": len(heldout), "curve": curve, "compute": __version__})


def emit(model: Any, s: Settings, folded: Mapping[str, np.ndarray], ref: Any, stream: Any,
         fidelity: Mapping[str, Any], *, trained: dict[str, Any]) -> dict[str, Any]:
    d_in, width = folded["w_enc"].shape
    d_out = int(folded["w_dec"].shape[1])
    model_id = read_model_id(model)
    space = {"model": model_id, "layer": s.layer, "point": s.point, "head": None, "d": int(d_in)}
    writer = ShardWriter(tempfile.mkdtemp(prefix="mechbench-tensor-"))
    encoders = np.ascontiguousarray(folded["w_enc"].T)
    norms = np.linalg.norm(folded["w_dec"].astype(np.float64), axis=1)
    for i in range(width):
        writer.add({"id": str(i), "index": i, "vector": folded["w_dec"][i], "encoder": encoders[i],
                    "norm": float(norms[i]), "b_enc": float(folded["b_enc"][i]),
                    "threshold": float(folded["threshold"][i])})
    activation: dict[str, Any] = {"fn": "jumprelu", "training": s.activation}
    activation.update({"k": s.k} if s.activation == "batchtopk" else {"l0": s.l0})
    return collection(
        "direction/dictionary", writer.close(),
        derivation="sae",
        reads=[space], writes=[{**space, "d": d_out}],
        model={"id": model_id, "architecture": getattr(getattr(model, "arch", None), "model_type", None)},
        width=int(width), d_in=int(d_in), d_out=d_out,
        activation=activation,
        b_dec=[float(x) for x in folded["b_dec"].astype(np.float64)],
        trained={"corpus": stream.describe(), "model": serialize_model(ref, model),
                 "hyperparameters": s.hyperparameters(), **trained},
        fidelity=fidelity,
    )
