from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.dictionaries.train_sae import param_names

CURVE_FIELDS = ("step", "tokens", "loss", "l0", "variance_explained", "dead")


def capture_state(trainer: Any, stream: Any, rng: np.random.Generator, *, scale: float, tokens: int,
                  heldout: np.ndarray, first_chunk: bytes, curve: list[dict[str, float]]) -> dict[str, Any]:
    opt: dict[str, Any] = {}
    for k in trainer.params:
        opt[f"m.{k}"] = np.array(trainer.m[k])
        opt[f"v.{k}"] = np.array(trainer.v[k])
    opt.update(stream.state())
    opt["fired_ago"] = trainer.fired_ago.copy()
    opt["heldout"] = np.asarray(heldout, dtype=np.int64)
    opt["first_chunk"] = np.frombuffer(first_chunk, dtype=np.uint8).copy()
    opt["curve"] = np.array([[p[f] for f in CURVE_FIELDS] for p in curve], dtype=np.float64).reshape(
        -1, len(CURVE_FIELDS))
    opt["scale"] = float(scale)
    opt["tokens"] = int(tokens)
    opt["threshold"] = float(trainer.threshold)
    return {"step": int(trainer.step),
            "weights": {k: np.array(v) for k, v in trainer.params.items()},
            "opt_state": opt,
            "np_rng": rng.bit_generator.state,
            "mx_key": None}


def scalar(value: Any) -> Any:
    return np.asarray(value).reshape(()).item()


def restore_state(trainer: Any, stream: Any, rng: np.random.Generator, state: Mapping[str, Any]) -> dict[str, Any]:
    weights = state["weights"]
    wanted = param_names(trainer.settings)
    if sorted(weights) != wanted:
        raise ValueError(f"dictionary/train: the saved training state holds {', '.join(sorted(weights))} and "
                         f"this node trains {', '.join(wanted)}: its `activation` differs from "
                         "the run that saved it; restart the node")
    opt = state["opt_state"]
    for k in trainer.params:
        trainer.params[k] = mx.array(np.asarray(weights[k], dtype=np.float32))
        trainer.m[k] = mx.array(np.asarray(opt[f"m.{k}"], dtype=np.float32))
        trainer.v[k] = mx.array(np.asarray(opt[f"v.{k}"], dtype=np.float32))
    mx.eval(trainer.params, trainer.m, trainer.v)
    trainer.step = int(state["step"])
    trainer.fired_ago = np.asarray(opt["fired_ago"], dtype=np.int64).copy()
    trainer.threshold = float(scalar(opt["threshold"]))
    stream.restore(opt)
    rng.bit_generator.state = state["np_rng"]
    curve = [dict(zip(CURVE_FIELDS, (float(v) for v in row), strict=True))
             for row in np.asarray(opt["curve"], dtype=np.float64).reshape(-1, len(CURVE_FIELDS))]
    for p in curve:
        p["step"], p["tokens"] = int(p["step"]), int(p["tokens"])
    return {"scale": float(scalar(opt["scale"])), "tokens": int(scalar(opt["tokens"])),
            "heldout": np.asarray(opt["heldout"], dtype=np.int64),
            "first_chunk": np.asarray(opt["first_chunk"], dtype=np.uint8).tobytes(), "curve": curve}
