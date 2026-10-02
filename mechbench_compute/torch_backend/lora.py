from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from mechbench_compute.adapter_keys import (
    ADAPTER_KEYS,
    AdapterKeys,
    check_layers,
    group_by_module,
)


def load_adapter_bytes(data: bytes) -> dict[str, Any]:
    from safetensors.torch import load

    return load(data)


def fuse(lm: Any, weights: Mapping[str, Any], scale: float, *, skip_missing: bool = False,
         skipped: list[str] | None = None, keys: AdapterKeys = ADAPTER_KEYS,
         layers: Sequence[int] | None = None) -> dict[tuple[int, str, str], Any]:
    import torch

    if layers is not None:
        check_layers(layers, len(lm.model.layers))
    pairs = group_by_module(weights, keys, layers)
    missing = [(i, c, p) for (i, c, p) in sorted(pairs)
               if not hasattr(getattr(lm.model.layers[i], c, None), p)]
    if missing and not skip_missing:
        by_proj: dict[str, list[int]] = {}
        for i, c, p in missing:
            by_proj.setdefault(f"{c}.{p}", []).append(i)
        detail = "; ".join(f"{k} on layers {v[0]}..{v[-1]} ({len(v)})" for k, v in by_proj.items())
        raise ValueError(
            f"adapter carries deltas for modules this architecture does not "
            f"expose: {detail}. It was trained under a different model "
            f"implementation. Pass adapter_skip_missing: true to fuse the "
            f"rest — the skipped modules are then reported on the result.")
    handle: dict[tuple[int, str, str], Any] = {}
    with torch.no_grad():
        for (i, container, proj), ab in sorted(pairs.items()):
            if set(ab) != {"a", "b"}:
                raise ValueError(
                    f"adapter is missing lora_a or lora_b for layer {i} {container}.{proj}")
            if (i, container, proj) in missing:
                if skipped is not None:
                    skipped.append(f"{i}.{container}.{proj}")
                continue
            mod = getattr(getattr(lm.model.layers[i], container), proj)
            held = mod.weight.data
            a, b = (ab["a"].to(held.device), ab["b"].to(held.device))
            handle[(i, container, proj)] = held
            mod.weight.data = held + (scale * (b @ a)).to(held.dtype)
    return handle


def restore(lm: Any, handle: Mapping[tuple[int, str, str], Any]) -> None:
    for (i, container, proj), held in handle.items():
        getattr(getattr(lm.model.layers[i], container), proj).weight.data = held


@contextlib.contextmanager
def unfused(lm: Any, handle: Mapping[tuple[int, str, str], Any]) -> Iterator[None]:
    fused = {(i, c, p): getattr(getattr(lm.model.layers[i], c), p).weight.data for i, c, p in handle}
    restore(lm, handle)
    try:
        yield
    finally:
        restore(lm, fused)


def fuse_adapter_stack(lm: Any, payloads: Sequence[Any], override_scale: float | None = None, *,
                       skip_missing: bool = False, skipped: list[str] | None = None,
                       keys: AdapterKeys = ADAPTER_KEYS,
                       layers: Sequence[Sequence[int] | None] = ()) -> list[dict]:
    from mechbench_compute.adapters.is_operator import is_operator

    handles: list[dict] = []
    try:
        for i, payload in enumerate(payloads):
            if is_operator(payload):
                raise ValueError(
                    "an operator attaches on the mlx backend only, and this model runs on "
                    "torch: fuse a LoRA here, or run the job on mlx")
            if not isinstance(payload, dict) or "data" not in payload:
                raise ValueError("adapter payload without safetensors bytes under 'data'")
            cfg = payload.get("lora") or {}
            scale = float(cfg.get("alpha", 16)) / float(cfg.get("rank", 8))
            if override_scale is not None and i == len(payloads) - 1:
                scale = float(override_scale)
            handles.append(fuse(lm, load_adapter_bytes(payload["data"]), scale,
                                skip_missing=skip_missing, skipped=skipped, keys=keys,
                                layers=layers[i] if i < len(layers) else None))
    except BaseException:
        restore_adapter_stack(lm, handles)
        raise
    return handles


def restore_adapter_stack(lm: Any, handles: Sequence[Mapping]) -> None:
    for handle in reversed(handles):
        restore(lm, handle)
