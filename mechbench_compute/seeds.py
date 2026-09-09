"""Seed derivation and the machine's identity (task 000402, epic
000364; DATAFLOW.md invariant 1).

Leaves have identity; chunks do not. Every mapped item's seed derives
from its KEY — root → member → item — never from its position in a
chunk or a batch, so a leaf's computation is the same wherever and in
whatever grouping it runs. The item rule is the one `generate` has
used since epic 000258 amendment 4, extracted verbatim: changing it
would change every corpus on the bench.

Hardware is recorded, never fingerprinted: chunks may move between
machines; lineage says where each ran, and the cross-machine contract
(bit-identical within a hardware class, envelope-bounded across) is
what a reader applies.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from typing import Any


def _digest_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def item_seed(seed: Any, record_id: Any, index: int) -> int:
    """The per-sample seed of `generate`: sha256("{seed}:{record}:{k}")[:8]."""
    return _digest_seed(f"{seed}:{record_id}:{index}")


def member_seed(root: Any, member_index: int) -> int:
    """A run-set member's seed from the root the parent designated."""
    return _digest_seed(f"member:{root}:{member_index}")


def derive(root: Any, *path: Any) -> int:
    """A seed for any key path under a root (chunk-independent)."""
    return _digest_seed(":".join([str(root), *(str(p) for p in path)]))


def hardware_class() -> dict[str, Any]:
    """What this machine is, for lineage: chip, GPU architecture,
    memory, MLX and Python versions, platform."""
    info: dict[str, Any] = {
        "platform": f"{sys.platform}/{platform.machine()}",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    try:
        import mlx.core as mx

        info["mlx"] = mx.__version__
        dev = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        info["chip"] = dev.get("device_name")
        info["gpu_arch"] = dev.get("architecture")
        info["memory_gb"] = round(int(dev.get("memory_size", 0)) / 2**30, 1)
    except Exception:  # noqa: BLE001 — no MLX (a pure runner): record what we can
        info["mlx"] = None
    return info


def hardware_class_id(info: dict[str, Any] | None = None) -> str:
    """The equivalence class under which bit-identity is promised:
    chip + GPU architecture + MLX version."""
    i = info or hardware_class()
    return f"{i.get('chip')}|{i.get('gpu_arch')}|mlx {i.get('mlx')}"
