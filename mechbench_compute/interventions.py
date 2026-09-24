from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import mlx.core as mx

from ._arch import E4B_DEFAULT
from .cache import ActivationCache
from .hooks import HookFn


@runtime_checkable
class Intervention(Protocol):
    def as_hooks(self) -> dict[str, HookFn]: ...
    def as_captures(self) -> list[str]: ...


def _zero(act: mx.array, info) -> mx.array:
    return mx.zeros_like(act)


@dataclass(frozen=True)
class _NamedZeroHook:
    names: tuple[str, ...]

    def as_hooks(self) -> dict[str, HookFn]:
        return {n: _zero for n in self.names}

    def as_captures(self) -> list[str]:
        return []


@dataclass(frozen=True)
class _Captures:
    names: tuple[str, ...]

    def as_hooks(self) -> dict[str, HookFn]:
        return {}

    def as_captures(self) -> list[str]:
        return list(self.names)


class _LayerAblation:
    __slots__ = ("layer_idx",)

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def as_hooks(self) -> dict[str, HookFn]:
        saved: dict[str, mx.array] = {}

        def cap(act, info):
            saved["v"] = act
            return None

        def restore(act, info):
            return saved["v"]

        return {
            f"blocks.{self.layer_idx}.resid_pre": cap,
            f"blocks.{self.layer_idx}.resid_post": restore,
        }

    def as_captures(self) -> list[str]:
        return []


class _HeadAblation:
    __slots__ = ("layer_idx", "head")

    def __init__(self, layer_idx: int, head: int):
        self.layer_idx = layer_idx
        self.head = head

    def as_hooks(self) -> dict[str, HookFn]:
        h = self.head

        def hook(act, info):
            n_heads = act.shape[1]
            mask = mx.ones((1, n_heads, 1, 1))
            mask = mask.at[:, h, :, :].add(-1.0)
            return act * mask

        return {f"blocks.{self.layer_idx}.attn.per_head_out": hook}

    def as_captures(self) -> list[str]:
        return []


class _PositionAdd:
    __slots__ = ("layer_idx", "position", "value", "alpha", "point")

    def __init__(self, layer_idx: int, position: int, value: mx.array,
                 alpha: float, point: str):
        self.layer_idx = layer_idx
        self.position = position
        self.value = value
        self.alpha = alpha
        self.point = point

    def as_hooks(self) -> dict[str, HookFn]:
        v = self.value
        pos = self.position
        alpha = self.alpha

        def hook(act, info):
            seq_len = act.shape[1]
            mask = mx.zeros((1, seq_len, 1), dtype=act.dtype)
            mask = mask.at[:, pos, :].add(1.0)
            return act + (alpha * v * mask)

        return {f"blocks.{self.layer_idx}.{self.point}": hook}

    def as_captures(self) -> list[str]:
        return []


class _PositionPatch:
    __slots__ = ("layer_idx", "position", "value", "point")

    def __init__(self, layer_idx: int, position: int, value: mx.array, point: str):
        self.layer_idx = layer_idx
        self.position = position
        self.value = value
        self.point = point

    def as_hooks(self) -> dict[str, HookFn]:
        v = self.value
        pos = self.position

        def hook(act, info):
            seq_len = act.shape[1]
            mask = mx.zeros((1, seq_len, 1), dtype=act.dtype)
            mask = mask.at[:, pos, :].add(1.0)
            return act * (1 - mask) + v * mask

        return {f"blocks.{self.layer_idx}.{self.point}": hook}

    def as_captures(self) -> list[str]:
        return []


def _norm_layers(layers) -> list[int]:
    if isinstance(layers, int):
        return [layers]
    return list(layers)


class Ablate:
    @staticmethod
    def layer(i: int) -> Intervention:
        return _LayerAblation(i)

    @staticmethod
    def attention(i: int) -> Intervention:
        return _NamedZeroHook(names=(f"blocks.{i}.attn_out",))

    @staticmethod
    def mlp(i: int) -> Intervention:
        return _NamedZeroHook(names=(f"blocks.{i}.mlp_out",))

    @staticmethod
    def head(layer: int, head: int) -> Intervention:
        return _HeadAblation(layer, head)

    @staticmethod
    def side_channel(layers: int | Iterable[int] | None = None) -> Intervention:
        if layers is None:
            layers = range(E4B_DEFAULT.n_layers)
        ls = _norm_layers(layers)
        return _NamedZeroHook(names=tuple(f"blocks.{i}.gate_out" for i in ls))


class Capture:
    @staticmethod
    def attn_weights(layers: int | Iterable[int]) -> Intervention:
        return _Captures(
            names=tuple(f"blocks.{i}.attn.weights" for i in _norm_layers(layers))
        )

    @staticmethod
    def residual(
        layers: int | Iterable[int], point: str = "post"
    ) -> Intervention:
        if point not in ("pre", "post"):
            raise ValueError(
                f"residual point must be 'pre' or 'post', got {point!r}"
            )
        return _Captures(
            names=tuple(
                f"blocks.{i}.resid_{point}" for i in _norm_layers(layers)
            )
        )

    @staticmethod
    def at(names: Iterable[str]) -> Intervention:
        return _Captures(names=tuple(str(n) for n in names))

    @staticmethod
    def final_norm_scale() -> Intervention:
        return _Captures(names=("final_norm.scale",))

    @staticmethod
    def gate_out(layers: int | Iterable[int]) -> Intervention:
        return _Captures(
            names=tuple(f"blocks.{i}.gate_out" for i in _norm_layers(layers))
        )

    @staticmethod
    def attn_out(layers: int | Iterable[int]) -> Intervention:
        return _Captures(
            names=tuple(f"blocks.{i}.attn_out" for i in _norm_layers(layers))
        )

    @staticmethod
    def mlp_out(layers: int | Iterable[int]) -> Intervention:
        return _Captures(
            names=tuple(f"blocks.{i}.mlp_out" for i in _norm_layers(layers))
        )

    @staticmethod
    def per_head_out(layers: int | Iterable[int]) -> Intervention:
        return _Captures(
            names=tuple(
                f"blocks.{i}.attn.per_head_out" for i in _norm_layers(layers)
            )
        )

    @staticmethod
    def queries(layers: int | Iterable[int]) -> Intervention:
        return _Captures(
            names=tuple(f"blocks.{i}.attn.q" for i in _norm_layers(layers))
        )

    @staticmethod
    def keys(layers: int | Iterable[int]) -> Intervention:
        return _Captures(
            names=tuple(f"blocks.{i}.attn.k" for i in _norm_layers(layers))
        )

    @staticmethod
    def values(layers: int | Iterable[int]) -> Intervention:
        return _Captures(
            names=tuple(f"blocks.{i}.attn.v" for i in _norm_layers(layers))
        )

    @staticmethod
    def qkv(layers: int | Iterable[int]) -> Intervention:
        names: list[str] = []
        for i in _norm_layers(layers):
            names.append(f"blocks.{i}.attn.q")
            names.append(f"blocks.{i}.attn.k")
            names.append(f"blocks.{i}.attn.v")
        return _Captures(names=tuple(names))


class Patch:
    @staticmethod
    def activation(
        layer: int,
        position: int,
        value: mx.array,
        *,
        point: str = "resid_post",
    ) -> Intervention:
        if point not in ("resid_pre", "resid_post"):
            raise ValueError(
                f"point must be 'resid_pre' or 'resid_post', got {point!r}"
            )
        return _PositionPatch(layer, position, value, point)

    @staticmethod
    def position(
        layer: int,
        position: int,
        source: ActivationCache,
        *,
        source_key: str | None = None,
        point: str = "resid_post",
    ) -> Intervention:
        key = source_key or f"blocks.{layer}.{point}"
        return Patch.activation(
            layer=layer, position=position, value=source[key], point=point,
        )

    @staticmethod
    def add(
        layer: int,
        position: int,
        value: mx.array,
        *,
        alpha: float = 1.0,
        point: str = "resid_post",
    ) -> Intervention:
        if point not in ("resid_pre", "resid_post"):
            raise ValueError(
                f"point must be 'resid_pre' or 'resid_post', got {point!r}"
            )
        return _PositionAdd(layer, position, value, float(alpha), point)


def _chain(fns: list[HookFn]) -> HookFn:
    def chained(act, info):
        for f in fns:
            new = f(act, info)
            if new is not None:
                act = new
        return act
    return chained


def compose(
    interventions: Iterable[Intervention] | None = None,
    *,
    hooks: dict[str, HookFn] | None = None,
    capture: Iterable[str] | None = None,
) -> tuple[dict[str, HookFn], list[str]]:
    by_point: dict[str, list[HookFn]] = {}
    captures: list[str] = []

    for iv in interventions or ():
        for name, fn in iv.as_hooks().items():
            by_point.setdefault(name, []).append(fn)
        captures.extend(iv.as_captures())

    for name, fn in (hooks or {}).items():
        by_point.setdefault(name, []).append(fn)
    captures.extend(capture or ())

    final_hooks: dict[str, HookFn] = {}
    for name, fns in by_point.items():
        final_hooks[name] = fns[0] if len(fns) == 1 else _chain(fns)
    return final_hooks, captures
