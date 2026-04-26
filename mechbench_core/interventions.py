"""Declarative interventions.

Each Intervention compiles to one or more hook callbacks plus zero or more
capture names. Pass a list of interventions to Model.run(interventions=[...])
and the framework composes them into the underlying hooks/capture call.

API:
    from mechbench_core import Model, Ablate, Capture, Patch

    model = Model.load()
    ids = model.tokenize("...")

    # Ablation: zero out a sub-component
    result = model.run(ids, interventions=[Ablate.mlp(14)])
    result = model.run(ids, interventions=[Ablate.head(29, head=7)])
    result = model.run(ids, interventions=[Ablate.side_channel()])  # all layers

    # Capture: snapshot tensors into result.cache
    result = model.run(ids, interventions=[
        Capture.attn_weights(layers=[23, 29]),
        Capture.residual(layers=range(42)),
    ])

    # Patch: replace an activation at one position with another
    clean = model.run(clean_ids).cache  # captured externally first
    result = model.run(corrupt_ids, interventions=[
        Capture.residual(layers=range(42)),  # for clean_ids
        Patch.position(layer=10, position=13, source=clean),
    ])

    # Combine freely
    result = model.run(ids, interventions=[
        Ablate.head(29, head=7),
        Capture.per_head_out(layers=[29]),  # captures the post-ablation tensor
    ])

Composition rule: when several callbacks target the same hook point, they
chain in declaration order. Each receives the previous return value and may
pass through (return None) or modify (return a new mx.array).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import mlx.core as mx

from ._arch import E4B_DEFAULT
from .cache import ActivationCache
from .hooks import HookFn


@runtime_checkable
class Intervention(Protocol):
    """Anything with `as_hooks()` and `as_captures()` is an Intervention.

    `as_hooks()` returns a dict mapping hook-point name -> callback. Called
    once per Model.run; build any required closure state inside the method
    so each invocation gets fresh state.

    `as_captures()` returns a list of hook-point names whose post-hook
    activation should be saved into the result's cache.
    """

    def as_hooks(self) -> dict[str, HookFn]: ...
    def as_captures(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# Internal helper interventions (frozen dataclasses for the simple cases)
# ---------------------------------------------------------------------------


def _zero(act: mx.array, info) -> mx.array:
    """Stateless 'zero this activation' hook callback."""
    return mx.zeros_like(act)


@dataclass(frozen=True)
class _NamedZeroHook:
    """Intervention that zeros the activation at one or more named hook points."""

    names: tuple[str, ...]

    def as_hooks(self) -> dict[str, HookFn]:
        return {n: _zero for n in self.names}

    def as_captures(self) -> list[str]:
        return []


@dataclass(frozen=True)
class _Captures:
    """Intervention that captures activations without modifying them."""

    names: tuple[str, ...]

    def as_hooks(self) -> dict[str, HookFn]:
        return {}

    def as_captures(self) -> list[str]:
        return list(self.names)


# ---------------------------------------------------------------------------
# Stateful interventions (need fresh closure per as_hooks() call)
# ---------------------------------------------------------------------------


class _LayerAblation:
    """Skip a layer entirely: residual stream passes through unchanged.

    Implementation: capture resid_pre at the start of the ablated layer,
    restore it at the end (overwriting whatever attn/MLP/gate/scalar produced).
    This bypasses the layer_scalar multiplication too, exactly matching the
    'skip the layer with continue' semantics of the original step_02 code.

    Each call to as_hooks() builds a fresh closure dict, so reusing the same
    Ablate.layer(i) instance across multiple Model.run calls is safe.
    """

    __slots__ = ("layer_idx",)

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def as_hooks(self) -> dict[str, HookFn]:
        saved: dict[str, mx.array] = {}

        def cap(act, info):
            saved["v"] = act
            return None  # don't modify on the way down

        def restore(act, info):
            return saved["v"]  # discard whatever h has become; revert

        return {
            f"blocks.{self.layer_idx}.resid_pre": cap,
            f"blocks.{self.layer_idx}.resid_post": restore,
        }

    def as_captures(self) -> list[str]:
        return []


class _HeadAblation:
    """Zero out one attention head's contribution at one layer.

    Hooks attn.per_head_out (shape [B, n_heads, L, head_dim]) and multiplies
    by a one-hot-zero mask. Forces the framework onto the manual attention
    path because attn.per_head_out is an internal point.
    """

    __slots__ = ("layer_idx", "head")

    def __init__(self, layer_idx: int, head: int):
        self.layer_idx = layer_idx
        self.head = head

    def as_hooks(self) -> dict[str, HookFn]:
        h = self.head

        def hook(act, info):
            # Match step_07's reference exactly: f32 mask (the default for
            # mx.ones without a dtype argument), promoting the multiplication
            # to f32 intermediate, downcast at the next op. Using act.dtype
            # (bf16) here would change intermediate rounding and produce
            # subtly different logits.
            n_heads = act.shape[1]
            mask = mx.ones((1, n_heads, 1, 1))
            mask = mask.at[:, h, :, :].add(-1.0)  # slot h becomes 0
            return act * mask

        return {f"blocks.{self.layer_idx}.attn.per_head_out": hook}

    def as_captures(self) -> list[str]:
        return []


class _PositionAdd:
    """Add a vector (scaled by alpha) to the activation at one (layer, position).

    The 'activation steering' / 'representation injection' operation common
    in interp work: a single steering vector v gets added at one position,
    leaving every other position untouched. v can be shape [d_model],
    [seq_len, d_model], or [1, seq_len, d_model] — broadcasting works.
    """

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
            # alpha * v * mask broadcasts to act's shape; nonzero only at pos
            return act + (alpha * v * mask)

        return {f"blocks.{self.layer_idx}.{self.point}": hook}

    def as_captures(self) -> list[str]:
        return []


class _PositionPatch:
    """Replace the activation at one (layer, position) with a fixed tensor.

    The tensor `value` is typically pulled from a clean-run ActivationCache
    (see Patch.position). Only the named token position is modified; the
    rest of the sequence is left untouched.
    """

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


# ---------------------------------------------------------------------------
# Public sugar: Ablate / Capture / Patch
# ---------------------------------------------------------------------------


def _norm_layers(layers) -> list[int]:
    if isinstance(layers, int):
        return [layers]
    return list(layers)


class Ablate:
    """Ablation interventions. Each .X(...) returns an Intervention.

    Replaces the six copy-pasted run_*_forward functions in the original
    experiment scripts: Ablate.layer (step_02), Ablate.attention/.mlp
    (step_04), Ablate.head (step_07), Ablate.side_channel (step_03).
    """

    @staticmethod
    def layer(i: int) -> Intervention:
        """Skip layer i entirely. Residual stream passes through unchanged.

        For non-KV-shared layers (0-23), attention still runs (populating
        the KV cache) so downstream KV-shared layers (24-41) can read it.
        Only the layer's CONTRIBUTION to the residual stream is discarded.
        """
        return _LayerAblation(i)

    @staticmethod
    def attention(i: int) -> Intervention:
        """Zero the attention branch's contribution at layer i. MLP and gate
        still run normally; KV cache is still populated."""
        return _NamedZeroHook(names=(f"blocks.{i}.attn_out",))

    @staticmethod
    def mlp(i: int) -> Intervention:
        """Zero the MLP branch's contribution at layer i. Attention and gate
        still run normally."""
        return _NamedZeroHook(names=(f"blocks.{i}.mlp_out",))

    @staticmethod
    def head(layer: int, head: int) -> Intervention:
        """Zero one attention head's slice of the per-head output at one layer.

        The head's keys/queries/values are still computed (and any KV-cache
        side effects still occur); only its contribution to the multi-head
        concatenation is discarded.
        """
        return _HeadAblation(layer, head)

    @staticmethod
    def side_channel(layers: int | Iterable[int] | None = None) -> Intervention:
        """Zero the MatFormer per-layer-input gate at the given layers.

        Pass None (the default) to ablate the side-channel everywhere — this
        is the catastrophic ablation from finding 03 that drops mean log p
        by ~30. Pass an int or iterable of layer indices for a more targeted
        ablation.
        """
        if layers is None:
            # side_channel is MatFormer-specific (Gemma 4 only); using the
            # E4B default here is intentional for the "ablate everywhere"
            # convenience. When called on a non-Gemma-4 variant, the
            # gate_out hooks won't exist and this will raise at run time
            # with a helpful CacheKeyError. Variant-aware validation lives
            # under task 000192.
            layers = range(E4B_DEFAULT.n_layers)
        ls = _norm_layers(layers)
        return _NamedZeroHook(names=tuple(f"blocks.{i}.gate_out" for i in ls))


class Capture:
    """Capture interventions. Each .X(...) returns an Intervention that adds
    the named hook points to the run's capture list.

    Captured tensors land in result.cache and can be retrieved by name. They
    are evaluated (mx.eval) before Model.run returns.
    """

    @staticmethod
    def attn_weights(layers: int | Iterable[int]) -> Intervention:
        """Capture post-softmax attention weights, shape [B, n_heads, L, S_kv].

        Forces the manual attention path. Used by step_05/06.
        """
        return _Captures(
            names=tuple(f"blocks.{i}.attn.weights" for i in _norm_layers(layers))
        )

    @staticmethod
    def residual(
        layers: int | Iterable[int], point: str = "post"
    ) -> Intervention:
        """Capture residual-stream values at the given layers.

        `point` is 'pre' (layer input) or 'post' (layer output). Default is
        'post', the more commonly useful endpoint and what step_01/08/10/11/12
        all use.
        """
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
    def gate_out(layers: int | Iterable[int]) -> Intervention:
        """Capture the MatFormer per-layer-input gate output at the given layers."""
        return _Captures(
            names=tuple(f"blocks.{i}.gate_out" for i in _norm_layers(layers))
        )

    @staticmethod
    def attn_out(layers: int | Iterable[int]) -> Intervention:
        """Capture the attention branch's contribution to the residual stream
        at the given layers, shape [B, L, d_model]. Post o_proj and
        post_attention_layernorm."""
        return _Captures(
            names=tuple(f"blocks.{i}.attn_out" for i in _norm_layers(layers))
        )

    @staticmethod
    def mlp_out(layers: int | Iterable[int]) -> Intervention:
        """Capture the MLP branch's contribution to the residual stream at the
        given layers, shape [B, L, d_model]. Post post_feedforward_layernorm."""
        return _Captures(
            names=tuple(f"blocks.{i}.mlp_out" for i in _norm_layers(layers))
        )

    @staticmethod
    def per_head_out(layers: int | Iterable[int]) -> Intervention:
        """Capture the per-head attention output BEFORE o_proj concatenation.

        Shape [B, n_heads, L, head_dim]. Forces the manual attention path.
        """
        return _Captures(
            names=tuple(
                f"blocks.{i}.attn.per_head_out" for i in _norm_layers(layers)
            )
        )

    @staticmethod
    def queries(layers: int | Iterable[int]) -> Intervention:
        """Capture per-head query tensors (post-q_norm + post-RoPE).

        Shape [B, n_heads, L, head_dim]. Forces the manual attention path.
        """
        return _Captures(
            names=tuple(f"blocks.{i}.attn.q" for i in _norm_layers(layers))
        )

    @staticmethod
    def keys(layers: int | Iterable[int]) -> Intervention:
        """Capture per-KV-head key tensors (post-k_norm + post-RoPE, pre-GQA-repeat).

        Shape [B, n_kv_heads, L_kv, head_dim]. Forces the manual attention path.
        n_kv_heads=2 for Gemma 4 E4B; each KV-head serves 4 Q-heads under the
        4:1 GQA ratio. In KV-shared layers (24-41) the keys come from the
        earlier layer's cache.
        """
        return _Captures(
            names=tuple(f"blocks.{i}.attn.k" for i in _norm_layers(layers))
        )

    @staticmethod
    def values(layers: int | Iterable[int]) -> Intervention:
        """Capture per-KV-head value tensors (post-v_norm, pre-GQA-repeat).

        Shape [B, n_kv_heads, L_kv, head_dim]. Forces the manual attention path.
        V does NOT get RoPE (only Q and K do).
        """
        return _Captures(
            names=tuple(f"blocks.{i}.attn.v" for i in _norm_layers(layers))
        )

    @staticmethod
    def qkv(layers: int | Iterable[int]) -> Intervention:
        """Capture Q, K, AND V per head for the given layers, in one intervention.

        Equivalent to composing Capture.queries + Capture.keys + Capture.values
        but shorter to write. Forces the manual attention path.
        """
        names: list[str] = []
        for i in _norm_layers(layers):
            names.append(f"blocks.{i}.attn.q")
            names.append(f"blocks.{i}.attn.k")
            names.append(f"blocks.{i}.attn.v")
        return _Captures(names=tuple(names))


class Patch:
    """Patch interventions. Each .X(...) returns an Intervention that
    replaces an activation at one (layer, position) with a supplied value.
    The classic use is causal tracing: capture activations from a clean run,
    then patch them into a corrupt run one (layer, position) at a time and
    measure the recovery."""

    @staticmethod
    def activation(
        layer: int,
        position: int,
        value: mx.array,
        *,
        point: str = "resid_post",
    ) -> Intervention:
        """Replace the activation at (layer, position) with `value`.

        `point` is 'resid_pre' or 'resid_post'. Default 'resid_post', matching
        step_09's forward_with_patch.
        """
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
        """Patch using a value pulled from `source` (a clean-run cache).

        Equivalent to Patch.activation(layer, position, value=source[key]).
        Defaults source_key to f'blocks.{layer}.{point}'.
        """
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
        """Add `alpha * value` to the activation at (layer, position).

        Sibling to Patch.activation, but additive instead of replacing.
        This is the canonical 'activation steering' / 'representation
        injection' operation: take a steering vector (typically a category
        centroid or a difference between two activations) and inject it
        into a different prompt's residual stream to test whether it
        steers the model's behavior.

        `value` shape can be [d_model], [seq_len, d_model], or
        [1, seq_len, d_model] — broadcasting works.

        `alpha` is the steering scale; alpha=0 is a no-op, alpha=1 adds
        the vector unchanged, larger values amplify.
        """
        if point not in ("resid_pre", "resid_post"):
            raise ValueError(
                f"point must be 'resid_pre' or 'resid_post', got {point!r}"
            )
        return _PositionAdd(layer, position, value, float(alpha), point)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _chain(fns: list[HookFn]) -> HookFn:
    """Combine a list of hook callbacks into a single chained callback.

    Each callback in the list receives the previous return value (or the
    original activation, for the first one) and may pass through (return
    None) or modify (return a new mx.array). Order = list order.
    """
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
    """Combine interventions, raw hooks, and raw captures into the
    (hooks_dict, capture_list) pair the forward pass expects.

    Composition rule:
      - Interventions are processed in iteration order; their hooks are
        registered first.
      - Raw `hooks` are registered next.
      - When several callbacks target the same hook point, they chain in
        registration order (earlier wraps later -> later wraps everything).
      - Captures are concatenated; duplicates are tolerated (the forward
        pass treats capture as a set).

    Used internally by Model.run when the user passes interventions=. Exposed
    as a public helper so power users can build hook dicts manually too.
    """
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
