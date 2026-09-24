from __future__ import annotations

import hashlib
import platform
import sys
from typing import Any


def _digest_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def item_seed(seed: Any, record_id: Any, index: int) -> int:
    return _digest_seed(f"{seed}:{record_id}:{index}")


def member_seed(root: Any, member_index: int) -> int:
    return _digest_seed(f"member:{root}:{member_index}")


def derive(root: Any, *path: Any) -> int:
    return _digest_seed(":".join([str(root), *(str(p) for p in path)]))


def hardware_class() -> dict[str, Any]:
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
    except Exception:  # noqa: BLE001
        info["mlx"] = None
    return info


def hardware_class_id(info: dict[str, Any] | None = None) -> str:
    i = info or hardware_class()
    return f"{i.get('chip')}|{i.get('gpu_arch')}|mlx {i.get('mlx')}"
