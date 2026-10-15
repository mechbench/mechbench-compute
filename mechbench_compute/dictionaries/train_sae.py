from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.dictionaries.train_settings import (
    ADAM_BETAS,
    ADAM_EPS,
    AUX_COEFFICIENT,
    THRESHOLD_DECAY,
    Settings,
)


def param_names(settings: Settings) -> list[str]:
    return sorted(["w_enc", "b_enc", "w_dec", "b_dec"] + (["log_threshold"] if settings.activation == "jumprelu" else []))


def init_params(settings: Settings, x: Any, seed: int) -> dict[str, Any]:
    d = int(x.shape[1])
    rng = np.random.default_rng([seed, 2])
    w_dec = rng.normal(size=(settings.width, d))
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    params = {"w_enc": mx.array(w_dec.T.astype(np.float32)),
              "b_enc": mx.zeros((settings.width,), dtype=mx.float32),
              "w_dec": mx.array(w_dec.astype(np.float32)),
              "b_dec": mx.mean(x, axis=0)}
    if settings.activation == "jumprelu":
        params["log_threshold"] = mx.full((settings.width,), math.log(settings.bandwidth(d)), dtype=mx.float32)
    mx.eval(params)
    return params


def make_jump(bandwidth: float):
    @mx.custom_function
    def jump(pre, threshold):
        return pre * (pre > threshold)

    @jump.vjp
    def jump_vjp(primals, cotangent, output):
        pre, threshold = primals
        near = (mx.abs(pre - threshold) < bandwidth / 2).astype(pre.dtype)
        return cotangent * (pre > threshold), mx.sum(cotangent * near, axis=0) * (-threshold / bandwidth)

    @mx.custom_function
    def step(pre, threshold):
        return (pre > threshold).astype(pre.dtype)

    @step.vjp
    def step_vjp(primals, cotangent, output):
        pre, threshold = primals
        near = (mx.abs(pre - threshold) < bandwidth / 2).astype(pre.dtype)
        return mx.zeros_like(pre), mx.sum(cotangent * near, axis=0) * (-1.0 / bandwidth)

    return jump, step


def batchtopk(pre: Any, k: int) -> tuple[Any, Any]:
    acts = mx.maximum(pre, 0)
    flat = acts.reshape(-1)
    keep = min(int(flat.size), k * int(pre.shape[0]))
    floor = mx.stop_gradient(mx.min(mx.topk(flat, keep)))
    mask = mx.stop_gradient((acts >= floor) & (acts > 0))
    return acts * mask, floor


def aux_loss(pre: Any, error: Any, dead: Any, w_dec: Any, k_aux: int) -> Any:
    acts = mx.maximum(pre, 0) * dead
    kth = mx.min(mx.topk(acts, k_aux, axis=-1), axis=-1, keepdims=True)
    f = acts * mx.stop_gradient((acts >= kth) & (acts > 0))
    return mx.mean(mx.sum((f @ w_dec - error) ** 2, axis=-1)) / error.shape[-1]


def make_loss(settings: Settings, d: int):
    jump, step = make_jump(settings.bandwidth(d))

    def loss_fn(params: Mapping[str, Any], x: Any, dead: Any, n_dead: int, sparsity: float):
        pre = (x - params["b_dec"]) @ params["w_enc"] + params["b_enc"]
        if settings.activation == "batchtopk":
            f, floor = batchtopk(pre, settings.k)
        else:
            threshold = mx.exp(params["log_threshold"])
            f, floor = jump(pre, threshold), mx.array(0.0)
        recon = f @ params["w_dec"] + params["b_dec"]
        mse = mx.mean(mx.sum((recon - x) ** 2, axis=-1)) / d
        loss = mse
        if settings.activation == "batchtopk" and n_dead:
            error = mx.stop_gradient(x - recon)
            loss = loss + AUX_COEFFICIENT * aux_loss(pre, error, dead, params["w_dec"], min(n_dead, settings.aux_k(d)))
        if settings.activation == "jumprelu":
            l0 = mx.mean(mx.sum(step(pre, mx.exp(params["log_threshold"])), axis=-1))
            loss = loss + sparsity * (l0 / settings.l0 - 1.0) ** 2
        return loss, mse, mx.stop_gradient(f), mx.stop_gradient(recon), floor

    return loss_fn


class Trainer:
    def __init__(self, settings: Settings, params: dict[str, Any], d: int) -> None:
        self.settings, self.params, self.d = settings, params, d
        self.m = {k: mx.zeros_like(v) for k, v in params.items()}
        self.v = {k: mx.zeros_like(v) for k, v in params.items()}
        self.step = 0
        self.fired_ago = np.zeros(settings.width, dtype=np.int64)
        self.threshold = -1.0
        self.grad = mx.value_and_grad(make_loss(settings, d))

    def train(self, x: Any) -> dict[str, float]:
        s = self.settings
        self.step += 1
        dead_np = self.fired_ago >= s.dead_tokens
        n_dead = int(dead_np.sum())
        dead = mx.array(dead_np.astype(np.float32))
        (loss, mse, f, recon, floor), grads = self.grad(self.params, x, dead, n_dead, s.sparsity_at(self.step))
        self.update(grads, s.lr_at(self.step))
        fired = np.array(mx.sum(f != 0, axis=0)) > 0
        active = float(mx.sum(f != 0).item())
        spread = float(mx.sum((x - mx.mean(x, axis=0)) ** 2).item())
        sse = float(mx.sum((recon - x) ** 2).item())
        n = int(x.shape[0])
        self.fired_ago += n
        self.fired_ago[fired] = 0
        if s.activation == "batchtopk":
            floor_v = float(floor.item())
            self.threshold = (floor_v if self.threshold < 0
                              else THRESHOLD_DECAY * self.threshold + (1 - THRESHOLD_DECAY) * floor_v)
        return {"step": self.step, "loss": float(loss.item()), "mse": float(mse.item()),
                "l0": active / n, "variance_explained": 1.0 - sse / spread if spread > 0 else 0.0,
                "dead": float((self.fired_ago >= s.dead_tokens).mean())}

    def update(self, grads: Mapping[str, Any], lr: float) -> None:
        b1, b2 = ADAM_BETAS
        t = self.step
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** t)
            v_hat = self.v[k] / (1 - b2 ** t)
            self.params[k] = self.params[k] - lr * m_hat / (mx.sqrt(v_hat) + ADAM_EPS)
        w = self.params["w_dec"]
        self.params["w_dec"] = w / mx.linalg.norm(w, axis=1, keepdims=True)
        mx.eval(self.params, self.m, self.v)

    def thresholds(self) -> np.ndarray:
        if self.settings.activation == "jumprelu":
            return np.exp(np.array(self.params["log_threshold"], dtype=np.float64))
        return np.full(self.settings.width, max(self.threshold, 0.0), dtype=np.float64)


def fold(params: Mapping[str, Any], thresholds: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    w_enc = np.array(params["w_enc"], dtype=np.float64)
    b_enc = np.array(params["b_enc"], dtype=np.float64)
    w_dec = np.array(params["w_dec"], dtype=np.float64)
    b_dec = np.array(params["b_dec"], dtype=np.float64)
    return {"w_enc": (w_enc * scale).astype(np.float32),
            "b_enc": (b_enc - b_dec @ w_enc).astype(np.float32),
            "threshold": thresholds.astype(np.float32),
            "w_dec": (w_dec / scale).astype(np.float32),
            "b_dec": (b_dec / scale).astype(np.float32)}
