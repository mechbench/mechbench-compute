from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

HEADROOM = 12 * 2**30
MEMINFO = Path("/proc/meminfo")


def read_index(device: Any) -> int:
    import torch

    return device.index if device.index is not None else torch.cuda.current_device()


def read_host_used(meminfo: Path = MEMINFO) -> int | None:
    try:
        fields = {}
        for line in meminfo.read_text().splitlines():
            name, _, rest = line.partition(":")
            if name in ("MemTotal", "MemAvailable"):
                fields[name] = int(rest.split()[0]) * 1024
        return fields["MemTotal"] - fields["MemAvailable"]
    except (OSError, ValueError, IndexError, KeyError):
        return None


def read_others_on_gpu(own: int | None = None) -> int:
    own = os.getpid() if own is None else own
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    total = 0
    for line in out.splitlines():
        pid, _, used = line.partition(",")
        try:
            if int(pid) != own:
                total += int(float(used)) * 2**20
        except ValueError:
            continue
    return total


# external: an integrated GPU (GB10) on Enverge's VMs — its memory is the machine's, the VM's /proc/meminfo never sees GPU allocations, and cudaMemGetInfo counts the host's reclaimable disk cache as used (59 GB "free" while 80 GB allocated cleanly), so the budget is the device's total less the VM's own use and other processes' GPU memory
def read_integrated_budget(index: int, meminfo: Path = MEMINFO) -> int | None:
    import torch

    host = read_host_used(meminfo)
    if host is None:
        return None
    total = torch.cuda.get_device_properties(index).total_memory
    return max(0, total - host - read_others_on_gpu())


def is_integrated(index: int) -> bool:
    import torch

    return bool(getattr(torch.cuda.get_device_properties(index), "is_integrated", False))


def read_free_bytes(device: Any) -> int | None:
    if getattr(device, "type", None) != "cuda":
        return None
    import torch

    index = read_index(device)
    if is_integrated(index):
        budget = read_integrated_budget(index)
        if budget is not None:
            return max(0, budget - torch.cuda.memory_reserved(index))
    free, _total = torch.cuda.mem_get_info(index)
    return int(free)


# external: the caching allocator on an integrated GPU (GB10) — it takes the machine's memory until the VM is reset instead of raising torch.OutOfMemoryError (Enverge's DGX Spark, 2026-10-12)
def cap_integrated(device: Any) -> float | None:
    if getattr(device, "type", None) != "cuda":
        return None
    import torch

    index = read_index(device)
    if not is_integrated(index):
        return None
    total = torch.cuda.get_device_properties(index).total_memory
    budget = read_integrated_budget(index)
    if budget is None:
        free, _ = torch.cuda.mem_get_info(index)
        budget = free + torch.cuda.memory_reserved(index)
    fraction = min(1.0, max(0.05, (budget - HEADROOM) / total))
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction
