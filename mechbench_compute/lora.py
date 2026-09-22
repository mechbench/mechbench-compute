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
from mlx import nn
from mlx.utils import tree_flatten

__all__ = [
    "LoRALinear",
    "apply_lora",
    "fuse",
    "load_adapter",
    "restore",
    "save_adapter",
]

KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([ab])$")

# Which submodule container each projection lives on. The set matches
# PEFT's target_modules: attention and MLP projections.
PROJ_CONTAINERS = {
    "q_proj": "self_attn", "k_proj": "self_attn",
    "v_proj": "self_attn", "o_proj": "self_attn",
    "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp",
}


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual:
    ``y = base(x) + (alpha/r) · (x @ Aᵀ) @ Bᵀ``, computed in fp32.
    ``B`` starts at zero so the wrapped model is exactly the base model
    at step 0.

    ``A`` is drawn from ``key`` when one is given, and from MLX's global
    generator otherwise. A key is what makes a training run repeatable:
    with ``B`` at zero the draw changes no output at step 0, but it
    changes every gradient after it, so without a key two runs of the
    same protocol with the same seed produce different adapters, by an
    envelope wider than the drift experiments compare releases by."""

    def __init__(self, base: nn.Module, r: int, alpha: float,
                 key: mx.array | None = None):
        super().__init__()
        self.base = base
        out_dim, in_dim = base.weight.shape
        self.scale = alpha / r
        draw = (mx.random.normal((r, in_dim), key=key) if key is not None
                else mx.random.normal((r, in_dim)))
        self.lora_a = draw * (1.0 / math.sqrt(in_dim))
        self.lora_b = mx.zeros((out_dim, r))

    def __call__(self, x):
        y = self.base(x)
        z = (x.astype(mx.float32) @ self.lora_a.T) @ self.lora_b.T
        return y + (self.scale * z).astype(y.dtype)


def apply_lora(lm, rank: int = 8, alpha: float = 16.0,
               targets: tuple[str, ...] = ("q_proj", "v_proj"),
               *, seed: int | None = None) -> int:
    """Freeze ``lm`` and wrap each named attention projection with a
    ``LoRALinear``, WHERE IT EXISTS. Returns the trainable parameter
    count.

    ``seed`` fixes the adapters' initial ``A`` matrices: each wrapped
    projection draws from its own subkey of ``mx.random.key(seed)``, in
    layer order, so the same model and targets give the same starting
    point every time and nothing else in the process is disturbed (the
    global generator is left alone). Without it the draw comes from the
    global generator and a training run is not repeatable — see
    ``LoRALinear``.

    "Where it exists" is not defensiveness — it is the architecture.
    gemma4's projections are conditional per layer: KV-shared tail
    layers have no k_proj/v_proj at all (they reuse an earlier layer's
    cache), and k-eq-v layers have no v_proj (K serves as V). Assuming
    a uniform tower here raises `'Attention' object has no attribute
    'v_proj'` on those variants. Skipped layers simply contribute no
    adapter weights; ``fuse`` walks the adapter's own keys, so sparse
    adapters round-trip untouched.

    A target that matches NO layer anywhere still refuses loudly —
    silently training nothing is the failure mode this function must
    never have."""
    lm.freeze()
    n = 0
    key = mx.random.key(seed) if seed is not None else None
    wrapped_per_target = dict.fromkeys(targets, 0)
    for layer in lm.model.layers:
        for name in targets:
            container = PROJ_CONTAINERS.get(name)
            if container is None:
                raise ValueError(f"unknown target module {name!r}; "
                                 f"known: {sorted(PROJ_CONTAINERS)}")
            holder = getattr(layer, container)
            base = getattr(holder, name, None)
            if base is None:
                continue
            sub = None
            if key is not None:
                key, sub = mx.random.split(key)
            wrapped = LoRALinear(base, rank, alpha, key=sub)
            setattr(holder, name, wrapped)
            wrapped_per_target[name] += 1
            n += wrapped.lora_a.size + wrapped.lora_b.size
    dead = [t for t, c in wrapped_per_target.items() if c == 0]
    if dead:
        raise ValueError(
            f"target modules {dead!r} exist on no layer of this "
            f"architecture — adapter would train nothing for them. "
            f"Wrapped counts: {wrapped_per_target!r}")
    return n


def save_adapter(lm, path: str) -> None:
    """Write the trainable (adapter) parameters to a safetensors file."""
    mx.save_safetensors(path, dict(tree_flatten(lm.trainable_parameters())))


def load_adapter(path: str) -> dict[str, mx.array]:
    return dict(mx.load(path))


def fuse(lm, weights: dict[str, mx.array],
         scale: float, *, skip_missing: bool = False,
         skipped: list[str] | None = None) -> dict[tuple[int, str], mx.array]:
    """Merge adapter deltas into ``lm``'s projection weights in place:
    ``W += scale · (B @ A)`` with ``scale = alpha / rank`` from training.

    Operates on the raw (unwrapped) modules of a fresh model. Returns a
    handle of the original weights; pass it to ``restore`` to undo the
    merge exactly (re-subtracting in low precision would not round-trip).

    An adapter may carry deltas for modules this architecture's current
    implementation does not expose — Gemma 4's KV-shared tail has no
    ``v_proj`` here, while an adapter trained under an implementation
    that did expose one carries a delta for every layer. A difference
    in the environment under the experiment must not be silent: by
    default it refuses, naming the modules. With ``skip_missing`` the
    applicable deltas fuse and every skipped module is appended to
    ``skipped`` (``"layer.container.proj"``), for the caller to report.
    """
    pairs: dict[tuple[int, str, str], dict[str, mx.array]] = {}
    for key, w in weights.items():
        m = KEY_RE.match(key)
        if m is None:
            raise ValueError(f"unrecognized adapter key {key!r}")
        i, container, proj, ab = (int(m.group(1)), m.group(2),
                                  m.group(3), m.group(4))
        pairs.setdefault((i, container, proj), {})[ab] = w
    missing = [(i, c, p) for (i, c, p) in sorted(pairs)
               if not hasattr(getattr(lm.model.layers[i], c, None), p)]
    if missing and not skip_missing:
        by_proj: dict[str, list[int]] = {}
        for i, c, p in missing:
            by_proj.setdefault(f"{c}.{p}", []).append(i)
        detail = "; ".join(f"{k} on layers {v[0]}..{v[-1]} ({len(v)})"
                           for k, v in by_proj.items())
        raise ValueError(
            f"adapter carries deltas for modules this architecture does not "
            f"expose: {detail}. It was trained under a different model "
            f"implementation. Pass adapter_skip_missing: true to fuse the "
            f"rest — the skipped modules are then reported on the result.")
    handle: dict[tuple[int, str, str], mx.array] = {}
    for (i, container, proj), ab in sorted(pairs.items()):
        if set(ab) != {"a", "b"}:
            raise ValueError(
                f"adapter is missing lora_a or lora_b for layer {i} "
                f"{container}.{proj}")
        if (i, container, proj) in missing:
            if skipped is not None:
                skipped.append(f"{i}.{container}.{proj}")
            continue
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


def fuse_adapter_stack(lm, payloads, override_scale=None, *,
                       skip_missing: bool = False,
                       skipped: list[str] | None = None):
    """Fuse an ORDERED adapter stack onto ``lm``.

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
            handles.append(fuse(lm, load_adapter(path), scale=scale,
                                skip_missing=skip_missing, skipped=skipped))
        finally:
            os.unlink(path)
    return handles


def restore_adapter_stack(lm, handles):
    """Undo ``fuse_adapter_stack``: reverse order, so each restore
    reinstalls the weights the NEXT-earlier fuse captured."""
    for handle in reversed(handles):
        restore(lm, handle)
