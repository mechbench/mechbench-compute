from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from mechbench_compute.intervene.compile_operator import compile_operator
from mechbench_compute.intervene.read_mask import Mask


def _write_polynomial(degree: int) -> str:
    return " + ".join(["c0", "c1 * x", *(f"c{i} * x ** {i}" for i in range(2, degree + 1))])


def _orthonormalize_rows(p: mx.array) -> mx.array:
    rows: list[mx.array] = []
    for i in range(p.shape[0]):
        v = p[i]
        for u in rows:
            v = v - mx.sum(v * u) * u
        rows.append(v / mx.sqrt(mx.sum(v * v)))
    return mx.stack(rows)


class OperatorModule(nn.Module):
    def __init__(self, spec: Mapping[str, Any], d: int, key: mx.array) -> None:
        super().__init__()
        self._d = int(d)
        self._positions = str(spec.get("positions") or "last")
        self._gated = bool(spec.get("gate", False))
        mask = spec.get("mask")
        self._rank = int(mask["rank"]) if isinstance(mask, Mapping) else 0
        self._mask = None if self._rank else Mask(dims=mask)
        self._function = str(spec["function"])
        self._degree = int(spec.get("degree") or 2)
        k = self._rank or (len(mask) if isinstance(mask, list) else self._d)
        first, second = mx.random.split(key)
        if self._rank:
            self.P = mx.random.normal((self._rank, self._d), key=first) / math.sqrt(self._d)
        if self._function == "affine" and self._rank:
            self.dW = mx.zeros((self._rank, self._d))
            self.b = mx.zeros((self._rank,))
        elif self._function == "affine":
            self.a = mx.ones((k,))
            self.b = mx.zeros((k,))
        elif self._function == "polynomial":
            self.c = mx.zeros((self._degree + 1, k)).at[1].add(1.0)
        else:
            width = int(spec.get("width") or 16)
            self.W1 = mx.random.normal((width, k), key=second) / math.sqrt(k)
            self.b1 = mx.zeros((width,))
            self.W2 = mx.zeros((k, width))
            self.b2 = mx.zeros((k,))
        if self._gated:
            self.gate = {"weight": mx.zeros((self._d,)), "bias": mx.zeros(())}

    def __call__(self, h: mx.array) -> mx.array:
        a = h.astype(mx.float32)
        new = self.edit(a)
        if self._gated:
            share = mx.sigmoid(a @ self.gate["weight"] + self.gate["bias"])[..., None]
            out = a + share * (new - a)
        elif self._positions == "last":
            last = (mx.arange(a.shape[-2]) == a.shape[-2] - 1)[:, None]
            out = mx.where(last, new, a)
        else:
            out = new
        return out.astype(h.dtype)

    def edit(self, a: mx.array) -> mx.array:
        if not self._rank:
            x = self._mask.read(a)
            return self._mask.write(a, x, self.evaluate(x), 1.0)
        r = _orthonormalize_rows(self.P)
        x = a @ r.T
        y = x + a @ self.dW.T + self.b if self._function == "affine" else self.evaluate(x)
        return a + (y - x) @ r

    def evaluate(self, x: mx.array) -> mx.array:
        if self._function == "affine":
            return compile_operator("a * x + b").evaluate({"x": x, "a": self.a, "b": self.b})
        if self._function == "polynomial":
            env = {f"c{i}": self.c[i] for i in range(self._degree + 1)}
            return compile_operator(_write_polynomial(self._degree)).evaluate({**env, "x": x})
        return x + mx.maximum(x @ self.W1.T + self.b1, 0.0) @ self.W2.T + self.b2

    def read_identity(self) -> dict[str, Any]:
        if self._function == "affine":
            return {"dW": 0.0, "b": 0.0} if self._rank else {"a": 1.0, "b": 0.0}
        if self._function == "polynomial":
            return {"c": mx.zeros_like(self.c).at[1].add(1.0)}
        return {"W2": 0.0, "b2": 0.0}

    def count_effective(self) -> int:
        moved = {name: np.array(getattr(self, name) - centre) != 0
                 for name, centre in self.read_identity().items()}
        count = sum(int(m.sum()) for m in moved.values())
        if self._function == "mlp":
            live = moved["W2"].any(axis=0)
            count += int(live.sum()) * (self.W1.shape[1] + 1)
        if self._rank and count:
            count += self.P.size
        if self._gated and count:
            count += self._d + 1
        return count

    def read_values(self) -> dict[str, Any]:
        from mlx.utils import tree_flatten

        return {name: np.array(value.astype(mx.float32)).tolist()
                for name, value in tree_flatten(self.parameters())}

    def load_values(self, values: Mapping[str, Any]) -> None:
        from mlx.utils import tree_flatten, tree_unflatten

        mine = dict(tree_flatten(self.parameters()))
        if set(values) != set(mine):
            raise ValueError(
                f"the operator's parameters are {sorted(values)}; its form has {sorted(mine)}")
        loaded = []
        for name, value in values.items():
            array = mx.array(np.asarray(value, dtype=np.float32))
            if tuple(array.shape) != tuple(mine[name].shape):
                raise ValueError(f"the operator's {name} is {tuple(array.shape)}; its form has "
                                 f"{tuple(mine[name].shape)}")
            loaded.append((name, array))
        self.update(tree_unflatten(loaded))
