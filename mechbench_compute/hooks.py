"""Hook-point types and name parsing.

A hook point is a named moment in the model's forward pass at which the
framework will (a) invoke any user-supplied callback and (b) optionally
record the activation into the result's cache. This module defines the
callback contract and the name-validation logic; the actual dispatch lives
in _forward.py.

The callback contract is intentionally TransformerLens-style: a function
taking (activation, info) that may return either None (leave unchanged) or
a replacement tensor. Returning a new tensor is how the user modifies
forward-pass behavior — e.g. zeroing it out for ablations, swapping in a
clean activation for causal tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx

from . import _arch
from .errors import InvalidHookName, LayerIndexOutOfRange


@dataclass(frozen=True)
class HookInfo:
    """Metadata passed to each hook callback alongside the activation tensor.

    Attributes:
        name: The full hook-point name, e.g. 'blocks.14.mlp_out'.
        layer: The layer index for layer-scoped hooks; None for top-level hooks.
        point: The point name within the layer, e.g. 'mlp_out'. For top-level
            hooks, equal to the full name.
    """

    name: str
    layer: Optional[int]
    point: str


# A hook callback. The function receives the activation tensor and a HookInfo
# describing which point it's at, and returns either None (leave unchanged)
# or an mx.array of identical shape and dtype that replaces it.
HookFn = Callable[[mx.array, HookInfo], Optional[mx.array]]


def parse_hook_name(name: str, arch: "_arch.Arch | None" = None) -> HookInfo:
    """Parse and validate a hook-point name.

    Accepts top-level names (those in _arch.GLOBAL_HOOK_POINTS) and layer-
    scoped names of the form 'blocks.{i}.{point}'. Raises InvalidHookName
    on unrecognized names; raises LayerIndexOutOfRange if the layer index
    is outside [0, arch.n_layers).

    Args:
        name: The hook-point name to validate.
        arch: The model's Arch (per-variant config). If None, defaults to
            the E4B layer count for backward compatibility with callers
            that haven't been ported to pass `model.arch`.
    """
    a = arch if arch is not None else _arch.E4B_DEFAULT

    if name in _arch.GLOBAL_HOOK_POINTS:
        return HookInfo(name=name, layer=None, point=name)

    if name.startswith("blocks."):
        rest = name[len("blocks."):]
        # Split on the first '.'; the point may contain further dots
        # (e.g. 'attn.weights'), which we don't want to split on.
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
                         arch: "_arch.Arch | None" = None) -> set[int]:
    """Return the set of layer indices at which the manual attention path
    must run because some hook or capture targets attention internals there.

    The manual softmax path is mathematically identical to the fused
    scaled_dot_product_attention kernel but produces slightly different bf16
    rounding. Switching paths per-layer (rather than globally) keeps the
    residual stream bitwise-equivalent at all layers where the user hasn't
    asked for attention internals — matching the existing experiment scripts'
    behavior of only using manual attention at the layers being inspected.
    """
    out: set[int] = set()
    for n in hook_names:
        info = parse_hook_name(n, arch=arch)
        if info.point in _arch.ATTN_INTERNAL_POINTS and info.layer is not None:
            out.add(info.layer)
    return out
