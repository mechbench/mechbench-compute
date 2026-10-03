from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, NamedTuple

SLOT = "operators"


class OperatorHandle(NamedTuple):
    attached: tuple[tuple[int, Any], ...]


@functools.cache
def _wrap_class(cls: type) -> type:
    class Attached(cls):
        def __call__(self, *args, **kwargs):
            out = super().__call__(*args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            for module in self[SLOT]:
                h = module(h)
            return (h, *out[1:]) if isinstance(out, tuple) else h

    Attached.__name__, Attached.__qualname__ = cls.__name__, cls.__qualname__
    Attached.unwrapped = cls
    return Attached


def attach_operators(lm, modules: Mapping[int, Any]) -> OperatorHandle:
    layers = lm.model.layers
    done: list[tuple[int, Any]] = []
    try:
        for i, module in sorted(modules.items()):
            layer = layers[i]
            if SLOT not in layer:
                layer[SLOT] = []
                layer.__class__ = _wrap_class(type(layer))
            layer[SLOT].append(module)
            done.append((i, module))
    except BaseException:
        detach_operators(lm, OperatorHandle(tuple(done)))
        raise
    return OperatorHandle(tuple(done))


def detach_operators(lm, handle: OperatorHandle) -> None:
    layers = lm.model.layers
    for i, module in reversed(handle.attached):
        layer = layers[i]
        held = layer.get(SLOT) or []
        kept = [m for m in held if m is not module]
        if kept:
            layer[SLOT] = kept
            continue
        layer.pop(SLOT, None)
        layer.__class__ = getattr(type(layer), "unwrapped", type(layer))
