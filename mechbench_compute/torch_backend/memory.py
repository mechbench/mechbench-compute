from __future__ import annotations

from pathlib import Path
from typing import Any

MEMINFO = Path("/proc/meminfo")


def read_host_available(meminfo: Path = MEMINFO) -> int | None:
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


# external: cudaMemGetInfo on an integrated GPU (GB10) — it reports the physical machine's free pages, not what this process can take (2-9 GB while 64 GB allocated on Enverge's VMs)
def read_free_bytes(device: Any) -> int | None:
    if getattr(device, "type", None) != "cuda":
        return None
    import torch

    index = device.index if device.index is not None else torch.cuda.current_device()
    if getattr(torch.cuda.get_device_properties(index), "is_integrated", False):
        available = read_host_available()
        if available is not None:
            return available
    free, _total = torch.cuda.mem_get_info(device)
    return int(free)
