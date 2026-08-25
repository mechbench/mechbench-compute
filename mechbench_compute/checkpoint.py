"""Checkpoints: merged models as first-class artifacts (000312 Arc C).

A checkpoint is a *directory* — safetensors shards, config, tokenizer —
published as a label PREFIX: one object per file plus a `manifest`
object naming every file with its size and sha256, and recording the
ModelRef the merge collapsed. The manifest is the provenance-bearing
parent; the shards are content-addressed leaves.

The merge itself is a DELTA-SHARD REWRITE, not a model round-trip: the
snapshot's `model.safetensors.index.json` maps tensor names to shards,
adapter keys name their layer/container/projection, so the merge copies
the snapshot and rewrites only the shards holding fused tensors
(``W += scale · (B @ A)``). No model is loaded; peak memory is one
shard. Untouched files stay byte-identical to the base — which the
manifest hashes make checkable.

Materialization is the inverse: fetch the manifest, fetch each file
into a cache directory keyed by the manifest's own content hash, verify
every file against its recorded sha256, and hand back a directory the
ordinary model loader can open. A half-fetched checkpoint can never be
loaded: files verify before the directory is marked complete.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx

from mechbench_compute.lora import PROJ_CONTAINERS

MANIFEST_NAME = "manifest"
_COMPLETE_MARK = ".complete"


def _adapter_deltas(payload: Mapping[str, Any]) -> dict[str, mx.array]:
    """One adapter payload -> {tensor_name: delta}, tensor names in the
    checkpoint's own naming (``...layers.N.<container>.<proj>.weight``,
    matched by suffix so leading scopes like `language_model.` never
    matter)."""
    import os
    import tempfile

    from mechbench_compute.lora import _KEY_RE, load_adapter

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

    pairs: dict[tuple[int, str, str], dict[str, mx.array]] = {}
    for key, w in weights.items():
        m = _KEY_RE.match(key)
        if m is None:
            raise ValueError(f"unrecognized adapter key {key!r}")
        i, container, proj, ab = (int(m.group(1)), m.group(2), m.group(3), m.group(4))
        if PROJ_CONTAINERS.get(proj) is None:
            raise ValueError(f"unknown projection {proj!r} in adapter")
        pairs.setdefault((i, container, proj), {})[ab] = w

    deltas: dict[str, mx.array] = {}
    for (i, container, proj), ab in pairs.items():
        if set(ab) != {"a", "b"}:
            raise ValueError(
                f"adapter is missing lora_a or lora_b for layer {i} "
                f"{container}.{proj}"
            )
        name_suffix = f"layers.{i}.{container}.{proj}.weight"
        deltas[name_suffix] = (scale * (ab["b"] @ ab["a"]))
    return deltas


def export_merged(
    snapshot_dir: str | Path,
    adapter_payloads: list[Mapping[str, Any]],
    out_dir: str | Path,
) -> list[str]:
    """Copy the snapshot, rewrite the shards the stack touches, in
    stack order (round two's delta lands on round one's result — the
    same left-to-right the fuse layer uses). Returns the file names
    written. Symlinked snapshots (the HF cache layout) are resolved to
    real bytes on copy."""
    snap = Path(snapshot_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = snap / "model.safetensors.index.json"
    if not index_path.exists():
        raise ValueError(
            f"{snap} has no model.safetensors.index.json — not a sharded "
            "checkpoint this merge knows how to rewrite"
        )
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    # suffix -> summed delta across the stack, applied in order.
    merged_deltas: dict[str, mx.array] = {}
    for payload in adapter_payloads:
        for suffix, delta in _adapter_deltas(payload).items():
            merged_deltas[suffix] = (
                merged_deltas[suffix] + delta if suffix in merged_deltas else delta
            )

    # Which shard holds each targeted tensor (full name resolved by
    # suffix match against the index).
    shard_targets: dict[str, dict[str, mx.array]] = {}
    for suffix, delta in merged_deltas.items():
        hits = [name for name in weight_map if name.endswith(suffix)]
        if len(hits) != 1:
            raise ValueError(
                f"tensor suffix {suffix!r} matched {len(hits)} entries in "
                f"the index — cannot merge safely"
            )
        shard_targets.setdefault(weight_map[hits[0]], {})[hits[0]] = delta

    written: list[str] = []
    for entry in sorted(snap.iterdir()):
        if entry.name.startswith("."):
            continue
        dest = out / entry.name
        if entry.name in shard_targets:
            # The HF cache stores snapshots as symlinks into an
            # extensionless blobs/ directory, and mx.load picks its
            # parser BY EXTENSION — so the link's own name must be what
            # it sees (resolve() handed it "blobs/ff4c28…", which is
            # "Unknown file format"). Both mx.load and copyfile follow
            # symlinks natively; nothing here needs resolve().
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
    """The provenance-bearing parent: every file with size and sha256,
    plus what this checkpoint IS — the ref it collapsed and the exact
    base snapshot it started from."""
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
        "kind": "checkpoint_manifest",
        "files": entries,
        "merged_from": dict(model_ref_wire),
        "base_snapshot": base_snapshot,
    }


def materialize(
    manifest: Mapping[str, Any],
    fetch_file: Callable[[str], Any],
    cache_root: str | Path,
) -> Path:
    """Fetch a checkpoint into the local cache and return its directory.

    `fetch_file(name)` yields the file's bytes (an iterator of chunks or
    a single bytes object). The directory is keyed by the manifest's own
    content hash, verified file-by-file, and marked complete only at the
    end — so a torn fetch is retried from scratch, never loaded.
    """
    key = hashlib.sha256(
        json.dumps(manifest.get("files"), sort_keys=True).encode()
    ).hexdigest()[:24]
    root = Path(cache_root)
    target = root / key
    mark = target / _COMPLETE_MARK
    if mark.exists():
        # A cache hit is a USE. The mark's mtime is what the runner's
        # eviction pass (000297) reads as last-used; without this touch
        # a checkpoint in weekly service looks as cold as its fetch date
        # and gets evicted into a 10 GB re-download.
        mark.touch()
        return target

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in manifest.get("files", []):
        name, want = str(entry["name"]), str(entry["sha256"])
        h = hashlib.sha256()
        with open(target / name, "wb") as f:
            data = fetch_file(name)
            chunks = [data] if isinstance(data, (bytes, bytearray)) else data
            for chunk in chunks:
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != want:
            shutil.rmtree(target)
            raise ValueError(
                f"checkpoint file {name!r} arrived with the wrong hash — "
                "refusing a corrupt materialization"
            )
    mark.touch()
    return target
