from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx

from . import _arch
from .errors import InvalidHookName, LayerIndexOutOfRange


@dataclass(frozen=True)
class HookInfo:
    name: str
    layer: Optional[int]
    point: str
    offset: int = 0


HookFn = Callable[[mx.array, HookInfo], Optional[mx.array]]


def parse_hook_name(name: str, arch: _arch.Arch | None = None) -> HookInfo:
    a = arch if arch is not None else _arch.E4B_DEFAULT

    if name in _arch.GLOBAL_HOOK_POINTS:
        return HookInfo(name=name, layer=None, point=name)

    if name.startswith("blocks."):
        rest = name[len("blocks."):]
        try:
            i_str, point = rest.split(".", 1)
            layer_idx = int(i_str)
        except ValueError:
            raise InvalidHookName(name, a.all_hook_names())

        if not (0 <= layer_idx < a.n_layers):
            raise LayerIndexOutOfRange(layer_idx, a.n_layers)

        if point not in _arch.LAYER_HOOK_POINTS:
            raise InvalidHookName(name, a.all_hook_names())

        return HookInfo(name=name, layer=layer_idx, point=point)

    raise InvalidHookName(name, a.all_hook_names())


def attn_internal_layers(hook_names: set[str],
                         arch: _arch.Arch | None = None) -> set[int]:
    out: set[int] = set()
    for n in hook_names:
        info = parse_hook_name(n, arch=arch)
        if info.point in _arch.ATTN_INTERNAL_POINTS and info.layer is not None:
            out.add(info.layer)
    return out


def mlp_internal_layers(hook_names: set[str],
                        arch: _arch.Arch | None = None) -> set[int]:
    out: set[int] = set()
    for n in hook_names:
        info = parse_hook_name(n, arch=arch)
        if info.point in _arch.MLP_INTERNAL_POINTS and info.layer is not None:
            out.add(info.layer)
    return out
