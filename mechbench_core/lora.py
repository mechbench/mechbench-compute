"""LoRA adapter machinery for training against ``Model.lm``.

Wraps attention projections with low-rank adapters (``apply_lora``),
saves the trained deltas (``save_adapter``), and merges saved deltas into
a fresh model's weights (``fuse``) with an exact undo (``restore``) so an
instrumented model can be flipped between base and adapted mid-script.

The freeze story matters: ``apply_lora`` freezes the *text decoder* it is
given — always pass ``Model.lm``, never the top-level multimodal module
(freezing a whole vlm walks non-module attributes on some audio/vision
towers and crashes). After freezing, only the ``lora_a``/``lora_b``
matrices are trainable, so ``lm.trainable_parameters()`` is exactly the
adapter and ``nn.value_and_grad(lm, loss)`` differentiates nothing else.

Adapter files are flat safetensors whose keys mirror the module tree
(``model.layers.{i}.self_attn.{proj}.lora_a`` / ``.lora_b``). ``fuse``
parses those keys, so it works for whatever projection set was trained.
The merge is ``W += scale · (B @ A)`` with ``scale = alpha / rank`` —
record the training alpha/rank with the adapter and pass the same scale.
"""

from __future__ import annotations

import math
import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

__all__ = [
    "LoRALinear",
    "apply_lora",
    "fuse",
    "load_adapter",
    "restore",
    "save_adapter",
]

_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.(\w+)\.lora_([ab])$")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual:
    ``y = base(x) + (alpha/r) · (x @ Aᵀ) @ Bᵀ``, computed in fp32.
    ``B`` starts at zero so the wrapped model is exactly the base model
    at step 0."""

    def __init__(self, base: nn.Module, r: int, alpha: float):
        super().__init__()
        self.base = base
        out_dim, in_dim = base.weight.shape
        self.scale = alpha / r
        self.lora_a = mx.random.normal((r, in_dim)) * (1.0 / math.sqrt(in_dim))
        self.lora_b = mx.zeros((out_dim, r))

    def __call__(self, x):
        y = self.base(x)
        z = (x.astype(mx.float32) @ self.lora_a.T) @ self.lora_b.T
        return y + (self.scale * z).astype(y.dtype)


def apply_lora(lm, rank: int = 8, alpha: float = 16.0,
               targets: tuple[str, ...] = ("q_proj", "v_proj")) -> int:
    """Freeze ``lm`` and wrap each named attention projection in every
    layer with a ``LoRALinear``. Returns the trainable parameter count."""
    lm.freeze()
    n = 0
    for layer in lm.model.layers:
        for name in targets:
            wrapped = LoRALinear(getattr(layer.self_attn, name), rank, alpha)
            setattr(layer.self_attn, name, wrapped)
            n += wrapped.lora_a.size + wrapped.lora_b.size
    return n


def save_adapter(lm, path: str) -> None:
    """Write the trainable (adapter) parameters to a safetensors file."""
    mx.save_safetensors(path, dict(tree_flatten(lm.trainable_parameters())))


def load_adapter(path: str) -> dict[str, mx.array]:
    return dict(mx.load(path))


def fuse(lm, weights: dict[str, mx.array],
         scale: float) -> dict[tuple[int, str], mx.array]:
    """Merge adapter deltas into ``lm``'s projection weights in place:
    ``W += scale · (B @ A)`` with ``scale = alpha / rank`` from training.

    Operates on the raw (unwrapped) modules of a fresh model. Returns a
    handle of the original weights; pass it to ``restore`` to undo the
    merge exactly (re-subtracting in low precision would not round-trip).
    """
    pairs: dict[tuple[int, str], dict[str, mx.array]] = {}
    for key, w in weights.items():
        m = _KEY_RE.match(key)
        if m is None:
            raise ValueError(f"unrecognized adapter key {key!r}")
        i, proj, ab = int(m.group(1)), m.group(2), m.group(3)
        pairs.setdefault((i, proj), {})[ab] = w
    handle: dict[tuple[int, str], mx.array] = {}
    for (i, proj), ab in sorted(pairs.items()):
        if set(ab) != {"a", "b"}:
            raise ValueError(
                f"adapter is missing lora_a or lora_b for layer {i} {proj}")
        mod = getattr(lm.model.layers[i].self_attn, proj)
        handle[(i, proj)] = mod.weight
        mod.weight = mod.weight + (scale * (ab["b"] @ ab["a"])).astype(
            mod.weight.dtype)
    mx.eval([getattr(lm.model.layers[i].self_attn, p).weight
             for i, p in handle])
    return handle


def restore(lm, handle: dict[tuple[int, str], mx.array]) -> None:
    """Undo a ``fuse`` by reinstalling the original weights."""
    for (i, proj), w in handle.items():
        getattr(lm.model.layers[i].self_attn, proj).weight = w
    mx.eval([getattr(lm.model.layers[i].self_attn, p).weight
             for i, p in handle])
