from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

from mechbench_compute.contained import check_file_name, write_inside
from mechbench_compute.lora import (
    ADAPTER_KEYS,
    AdapterKeys,
    check_layers,
    group_by_module,
)

MANIFEST_NAME = "manifest"
_COMPLETE_MARK = ".complete"


def _adapter_deltas(payload: Mapping[str, Any], keys: AdapterKeys = ADAPTER_KEYS,
                    layers: Sequence[int] | None = None) -> dict[str, mx.array]:
    import os
    import tempfile

    from mechbench_compute.lora import load_adapter

    cfg = payload.get("lora") or {}
    scale = float(cfg.get("alpha", 16)) / float(cfg.get("rank", 8))

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(payload["data"])
        weights = load_adapter(path)
    finally:
        os.unlink(path)

    deltas: dict[str, mx.array] = {}
    for (i, container, proj), ab in group_by_module(weights, keys, layers).items():
        if keys.containers.get(proj) is None:
            raise ValueError(f"unknown projection {proj!r} in adapter")
        if set(ab) != {"a", "b"}:
            raise ValueError(
                f"adapter is missing lora_a or lora_b for layer {i} "
                f"{container}.{proj}"
            )
        name_suffix = f"layers.{i}.{container}.{proj}.weight"
        deltas[name_suffix] = (scale * (ab["b"] @ ab["a"]))
    return deltas


def read_adapter_keys(snap: Path, layers: Sequence[Sequence[int] | None] = ()) -> AdapterKeys:
    from mechbench_compute._arch import read_arch_from_config
    from mechbench_compute.architectures import for_type

    config_path = snap / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    declared = for_type(config.get("model_type"))
    chosen = [one for one in layers if one is not None]
    if chosen:
        if declared is None:
            raise ValueError(
                f"an adapter's `layers` is checked against the checkpoint's layers, and its "
                f"config.json names model_type {config.get('model_type')!r}, which compute "
                f"does not load")
        n_layers = read_arch_from_config(config).n_layers
        for one in chosen:
            check_layers(one, n_layers)
    return ADAPTER_KEYS if declared is None else declared.adapter_keys


def export_merged(
    snapshot_dir: str | Path,
    adapter_payloads: list[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    layers: Sequence[Sequence[int] | None] = (),
) -> list[str]:
    from mechbench_compute.adapters.is_operator import is_operator

    snap = Path(snapshot_dir)
    out = Path(out_dir)
    if any(is_operator(p) for p in adapter_payloads):
        raise ValueError(
            "an operator acts on activations and a checkpoint holds weights, so it cannot be "
            "merged: merge the LoRA adapters, and keep the operator in the model reference")

    index_path = snap / "model.safetensors.index.json"
    if not index_path.exists():
        raise ValueError(
            f"{snap} has no model.safetensors.index.json — not a sharded "
            "checkpoint this merge knows how to rewrite"
        )
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    keys = read_adapter_keys(snap, layers)

    merged_deltas: dict[str, mx.array] = {}
    for index, payload in enumerate(adapter_payloads):
        chosen = layers[index] if index < len(layers) else None
        for suffix, delta in _adapter_deltas(payload, keys, chosen).items():
            merged_deltas[suffix] = (
                merged_deltas[suffix] + delta if suffix in merged_deltas else delta
            )

    shard_targets: dict[str, dict[str, mx.array]] = {}
    for suffix, delta in merged_deltas.items():
        hits = [name for name in weight_map if name.endswith(suffix)]
        if len(hits) != 1:
            raise ValueError(
                f"tensor suffix {suffix!r} matched {len(hits)} entries in "
                f"the index — cannot merge safely"
            )
        shard_targets.setdefault(weight_map[hits[0]], {})[hits[0]] = delta

    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for entry in sorted(snap.iterdir()):
        if entry.name.startswith("."):
            continue
        dest = out / entry.name
        if entry.name in shard_targets:
            tensors = dict(mx.load(str(entry)))
            for name, delta in shard_targets[entry.name].items():
                w = tensors[name]
                tensors[name] = (w + delta.astype(w.dtype)).astype(w.dtype)
            mx.eval(list(tensors.values()))
            mx.save_safetensors(str(dest), tensors)
        else:
            shutil.copyfile(entry, dest)
        written.append(entry.name)
    return written


def build_manifest(
    out_dir: str | Path,
    files: list[str],
    model_ref_wire: Mapping[str, Any],
    base_snapshot: str,
) -> dict[str, Any]:
    out = Path(out_dir)
    entries = []
    for name in sorted(files):
        p = out / name
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        entries.append(
            {"name": name, "size": p.stat().st_size, "sha256": h.hexdigest()}
        )
    return {
        "kind": "adapter/checkpoint",
        "files": entries,
        "merged_from": dict(model_ref_wire),
        "base_snapshot": base_snapshot,
    }


def materialize(
    manifest: Mapping[str, Any],
    fetch_file: Callable[[str], Any],
    cache_root: str | Path,
    on_bytes: Callable[[int, int], None] | None = None,
) -> Path:
    files = list(manifest.get("files") or [])
    names = [check_file_name(e.get("name"), what="checkpoint file") for e in files]
    key = hashlib.sha256(
        json.dumps(manifest.get("files"), sort_keys=True).encode()
    ).hexdigest()[:24]
    root = Path(cache_root)
    target = root / key
    mark = target / _COMPLETE_MARK
    if mark.exists():
        mark.touch()
        return target

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    total = sum(int(e.get("size", 0)) for e in files)
    done = 0
    last_reported = 0

    def count(n: int) -> None:
        nonlocal done, last_reported
        done += n
        if on_bytes is not None and done - last_reported >= (4 << 20):
            last_reported = done
            on_bytes(done, total)

    for entry, name in zip(files, names, strict=True):
        data = fetch_file(name)
        chunks = [data] if isinstance(data, (bytes, bytearray)) else data
        ok = write_inside(target, name, chunks, want=str(entry["sha256"]),
                          on_chunk=count)
        if on_bytes is not None:
            last_reported = done
            on_bytes(done, total)
        if not ok:
            shutil.rmtree(target)
            raise ValueError(
                f"checkpoint file {name!r} arrived with the wrong hash — "
                "refusing a corrupt materialization"
            )
    mark.touch()
    return target
