from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .adapter_keys import ADAPTER_KEYS, AdapterKeys, check_layers, group_by_module

__all__ = [
    "LoRALinear",
    "apply_lora",
    "fuse",
    "load_adapter",
    "restore",
    "save_adapter",
    "unfused",
]


class LoRALinear(nn.Module):
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


# external: mlx-vlm — freezing a whole VLM walks non-module attributes of some audio/vision towers and crashes; pass Model.lm
def apply_lora(lm, rank: int = 8, alpha: float = 16.0,
               targets: tuple[str, ...] = ("q_proj", "v_proj"),
               *, seed: int | None = None, keys: AdapterKeys = ADAPTER_KEYS) -> int:
    lm.freeze()
    n = 0
    key = mx.random.key(seed) if seed is not None else None
    wrapped_per_target = dict.fromkeys(targets, 0)
    try:
        for layer in lm.model.layers:
            for name in targets:
                container = keys.containers.get(name)
                if container is None:
                    raise ValueError(f"unknown target module {name!r}; "
                                     f"known: {sorted(keys.containers)}")
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
    except BaseException:
        remove_lora(lm)
        raise
    return n


def remove_lora(lm) -> int:
    removed = 0
    for _, module in lm.named_modules():
        for name, child in list(module.items()):
            if isinstance(child, LoRALinear):
                module[name] = child.base
                removed += 1
    return removed


def mark_weights(lm) -> dict[str, mx.array]:
    return dict(tree_flatten(lm.parameters()))


def read_changed(lm, marks) -> list[str]:
    now = mark_weights(lm)
    return sorted(name for name in now.keys() | marks.keys()
                  if now.get(name) is not marks.get(name))


def save_adapter(lm, path: str) -> None:
    mx.save_safetensors(path, dict(tree_flatten(lm.trainable_parameters())))


def read_adapter_bytes(lm) -> bytes:
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        save_adapter(lm, path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def load_adapter(path: str) -> dict[str, mx.array]:
    return dict(mx.load(path))




def fuse(lm, weights: dict[str, mx.array],
         scale: float, *, skip_missing: bool = False,
         skipped: list[str] | None = None,
         keys: AdapterKeys = ADAPTER_KEYS,
         layers: Sequence[int] | None = None) -> dict[tuple[int, str], mx.array]:
    if layers is not None:
        check_layers(layers, len(lm.model.layers))
    pairs = group_by_module(weights, keys, layers)
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
    try:
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
    except BaseException:
        restore(lm, handle)
        raise
    return handle


def restore(lm, handle: dict[tuple[int, str, str], mx.array]) -> None:
    for (i, container, proj), w in handle.items():
        getattr(getattr(lm.model.layers[i], container), proj).weight = w
    mx.eval([getattr(getattr(lm.model.layers[i], c), p).weight
             for i, c, p in handle])


@contextlib.contextmanager
def unfused(lm, handle: dict[tuple[int, str, str], mx.array]) -> Iterator[None]:
    fused = {(i, c, p): getattr(getattr(lm.model.layers[i], c), p).weight for i, c, p in handle}
    restore(lm, handle)
    try:
        yield
    finally:
        restore(lm, fused)


def fuse_adapter_stack(lm, payloads, override_scale=None, *,
                       skip_missing: bool = False,
                       skipped: list[str] | None = None,
                       keys: AdapterKeys = ADAPTER_KEYS,
                       layers: Sequence[Sequence[int] | None] = ()):
    import os
    import tempfile

    from mechbench_compute.adapters.attach_payload import attach_payload
    from mechbench_compute.adapters.is_operator import is_operator

    handles = []
    try:
        for i, payload in enumerate(payloads):
            if is_operator(payload):
                if i < len(layers) and layers[i] is not None:
                    raise ValueError(
                        "an operator acts at the layers it was trained at; `layers` belongs to "
                        "a LoRA: leave it out of the operator's entry")
                handles.append(attach_payload(lm, payload))
                continue
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
                                    skip_missing=skip_missing, skipped=skipped, keys=keys,
                                    layers=layers[i] if i < len(layers) else None))
            finally:
                os.unlink(path)
    except BaseException:
        restore_adapter_stack(lm, handles)
        raise
    return handles


def restore_adapter_stack(lm, handles):
    from mechbench_compute.adapters.attach_operators import (
        OperatorHandle,
        detach_operators,
    )

    for handle in reversed(handles):
        if isinstance(handle, OperatorHandle):
            detach_operators(lm, handle)
        else:
            restore(lm, handle)
