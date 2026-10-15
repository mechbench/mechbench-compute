from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import torch

from mechbench_compute.adapters.read_operator_spec import write_polynomial
from mechbench_compute.intervene.compile_operator import compile_operator
from mechbench_compute.intervene.read_mask import Mask
from mechbench_compute.torch_backend.mlx_random import (
    draw_normal,
    make_key,
    split_key,
    split_keys,
)


def orthonormalize_rows(p: torch.Tensor) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for i in range(p.shape[0]):
        v = p[i]
        for u in rows:
            v = v - torch.sum(v * u) * u
        rows.append(v / torch.sqrt(torch.sum(v * v)))
    return torch.stack(rows)


class TorchOperator:
    def __init__(self, spec: Mapping[str, Any], d: int, key: np.ndarray, device: Any) -> None:
        self.d = int(d)
        self.positions = str(spec.get("positions") or "last")
        self.gated = bool(spec.get("gate", False))
        mask = spec.get("mask")
        self.rank = int(mask["rank"]) if isinstance(mask, Mapping) else 0
        self.mask = None if self.rank else Mask(dims=mask)
        self.function = str(spec["function"])
        self.degree = int(spec.get("degree") or 2)
        self.params: dict[str, torch.nn.Parameter] = {}
        k = self.k = self.rank or (len(mask) if isinstance(mask, list) else self.d)
        first, second = split_key(key)

        def put(name: str, value: np.ndarray) -> None:
            self.params[name] = torch.nn.Parameter(
                torch.from_numpy(np.asarray(value, dtype=np.float32).copy()).to(device))

        if self.rank:
            put("P", draw_normal((self.rank, self.d), first) / np.float32(math.sqrt(self.d)))
        if self.function == "affine" and self.rank:
            put("dW", np.zeros((self.rank, self.d)))
            put("b", np.zeros((self.rank,)))
        elif self.function == "affine":
            put("a", np.ones((k,)))
            put("b", np.zeros((k,)))
        elif self.function == "polynomial":
            put("c", self.read_identity()["c"])
        else:
            width = int(spec.get("width") or 16)
            put("W1", draw_normal((width, k), second) / np.float32(math.sqrt(k)))
            put("b1", np.zeros((width,)))
            put("W2", np.zeros((k, width)))
            put("b2", np.zeros((k,)))
        if self.gated:
            put("gate.weight", np.zeros((self.d,)))
            put("gate.bias", np.zeros(()))

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        a = h.float()
        new = self.edit(a)
        if self.gated:
            share = torch.sigmoid(a @ self.params["gate.weight"] + self.params["gate.bias"])[..., None]
            out = a + share * (new - a)
        elif self.positions == "last":
            last = (torch.arange(a.shape[-2], device=a.device) == a.shape[-2] - 1)[:, None]
            out = torch.where(last, new, a)
        else:
            out = new
        return out.to(h.dtype)

    def edit(self, a: torch.Tensor) -> torch.Tensor:
        if not self.rank:
            x = self.mask.read(a)
            return self.mask.write(a, x, self.evaluate(x), 1.0)
        r = orthonormalize_rows(self.params["P"])
        x = a @ r.T
        y = (x + a @ self.params["dW"].T + self.params["b"] if self.function == "affine"
             else self.evaluate(x))
        return a + (y - x) @ r

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        p = self.params
        if self.function == "affine":
            return compile_operator("a * x + b").evaluate({"x": x, "a": p["a"], "b": p["b"]})
        if self.function == "polynomial":
            env = {f"c{i}": p["c"][i] for i in range(self.degree + 1)}
            return compile_operator(write_polynomial(self.degree)).evaluate({**env, "x": x})
        return x + torch.relu(x @ p["W1"].T + p["b1"]) @ p["W2"].T + p["b2"]

    def read_identity(self) -> dict[str, Any]:
        if self.function == "affine":
            return {"dW": 0.0, "b": 0.0} if self.rank else {"a": 1.0, "b": 0.0}
        if self.function == "polynomial":
            c = np.zeros((self.degree + 1, self.k), dtype=np.float32)
            c[1] = 1.0
            return {"c": c}
        return {"W2": 0.0, "b2": 0.0}

    def count_effective(self) -> int:
        moved = {name: (self.params[name].detach().cpu().numpy() - centre) != 0
                 for name, centre in self.read_identity().items()}
        count = sum(int(m.sum()) for m in moved.values())
        if self.function == "mlp":
            count += int(moved["W2"].any(axis=0).sum()) * (self.params["W1"].shape[1] + 1)
        if self.rank and count:
            count += self.params["P"].numel()
        if self.gated and count:
            count += self.d + 1
        return count

    def read_values(self) -> dict[str, Any]:
        return {name: p.detach().float().cpu().numpy().tolist() for name, p in self.params.items()}


class OperatorHandle(NamedTuple):
    operators: dict[int, TorchOperator]
    removers: tuple[Any, ...]


def build_operators(spec: Mapping[str, Any], d: int, *, seed: int, device: Any) -> dict[int, TorchOperator]:
    keys = split_keys(make_key(int(seed)), len(spec["layers"]))
    return {int(layer): TorchOperator(spec, d, key, device) for layer, key in zip(spec["layers"], keys)}


def read_operator_parameters(operators: Mapping[int, TorchOperator]) -> dict[str, torch.nn.Parameter]:
    return {f"model.layers.{i}.operators.0.{name}": p
            for i, op in sorted(operators.items()) for name, p in op.params.items()}


def attach_operators(lm: Any, operators: Mapping[int, TorchOperator]) -> OperatorHandle:
    removers = []
    try:
        for i, op in sorted(operators.items()):
            removers.append(lm.model.layers[i].register_forward_hook(make_hook(op)))
    except BaseException:
        for r in removers:
            r.remove()
        raise
    return OperatorHandle(dict(operators), tuple(removers))


def make_hook(op: TorchOperator) -> Any:
    def operate(module: Any, args: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            return (op(output[0]), *output[1:])
        return op(output)

    return operate


def detach_operators(handle: OperatorHandle) -> None:
    for r in reversed(handle.removers):
        r.remove()


@torch.no_grad()
def prune_operators(operators: Mapping[int, TorchOperator], opt: Any, l1: float) -> None:
    for i, op in sorted(operators.items()):
        for name, centre in op.read_identity().items():
            p = op.params[name]
            v = opt.v[f"model.layers.{i}.operators.0.{name}"]
            room = l1 * opt.lr / (torch.sqrt(v) + opt.eps)
            centre = torch.as_tensor(centre, dtype=p.dtype, device=p.device)
            moved = p - centre
            p.copy_(centre + torch.sign(moved) * torch.clamp(torch.abs(moved) - room, min=0.0))
