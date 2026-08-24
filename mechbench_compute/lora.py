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
    r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([ab])$")

# Which submodule container each projection lives on (000263: PEFT's
# target_modules generality — attention and MLP projections).
PROJ_CONTAINERS = {
    "q_proj": "self_attn", "k_proj": "self_attn",
    "v_proj": "self_attn", "o_proj": "self_attn",
    "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp",
}


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
            container = PROJ_CONTAINERS.get(name)
            if container is None:
                raise ValueError(f"unknown target module {name!r}; "
                                 f"known: {sorted(PROJ_CONTAINERS)}")
            holder = getattr(layer, container)
            wrapped = LoRALinear(getattr(holder, name), rank, alpha)
            setattr(holder, name, wrapped)
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
    pairs: dict[tuple[int, str, str], dict[str, mx.array]] = {}
    for key, w in weights.items():
        m = _KEY_RE.match(key)
        if m is None:
            raise ValueError(f"unrecognized adapter key {key!r}")
        i, container, proj, ab = (int(m.group(1)), m.group(2),
                                  m.group(3), m.group(4))
        pairs.setdefault((i, container, proj), {})[ab] = w
    handle: dict[tuple[int, str, str], mx.array] = {}
    for (i, container, proj), ab in sorted(pairs.items()):
        if set(ab) != {"a", "b"}:
            raise ValueError(
                f"adapter is missing lora_a or lora_b for layer {i} "
                f"{container}.{proj}")
        mod = getattr(getattr(lm.model.layers[i], container), proj)
        handle[(i, container, proj)] = mod.weight
        mod.weight = mod.weight + (scale * (ab["b"] @ ab["a"])).astype(
            mod.weight.dtype)
    mx.eval([getattr(getattr(lm.model.layers[i], c), p).weight
             for i, c, p in handle])
    return handle


def restore(lm, handle: dict[tuple[int, str, str], mx.array]) -> None:
    """Undo a ``fuse`` by reinstalling the original weights."""
    for (i, container, proj), w in handle.items():
        getattr(getattr(lm.model.layers[i], container), proj).weight = w
    mx.eval([getattr(getattr(lm.model.layers[i], c), p).weight
             for i, c, p in handle])


def fuse_adapter_stack(lm, payloads, override_scale=None):
    """Fuse an ORDERED adapter stack onto ``lm`` (task 000312 Arc B).

    Successive fine-tuning rounds compose by fusing left to right: each
    fuse's restore handle captures the weights as the previous rounds
    left them, so the returned handles undo cleanly ONLY in reverse
    order — which is what ``restore_adapter_stack`` does, and why the
    handles come back as a list rather than a merged dict (a later
    adapter may touch projections an earlier one did not).

    Each payload carries its own scale (alpha/rank from its training);
    ``override_scale`` applies to the LAST payload only — it is the
    single-adapter knob (params.adapter_scale) and the last position is
    the node's own operand.
    """
    import os
    import tempfile

    handles = []
    for i, payload in enumerate(payloads):
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError(
                "adapter payload without safetensors bytes under 'data'")
        cfg = payload.get("lora") or {}
        scale = float(cfg.get("alpha", 16)) / float(cfg.get("rank", 8))
        if override_scale is not None and i == len(payloads) - 1:
            scale = float(override_scale)
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(payload["data"])
            handles.append(fuse(lm, load_adapter(path), scale=scale))
        finally:
            os.unlink(path)
    return handles


def restore_adapter_stack(lm, handles):
    """Undo ``fuse_adapter_stack``: reverse order, so each restore
    reinstalls the weights the NEXT-earlier fuse captured."""
    for handle in reversed(handles):
        restore(lm, handle)
