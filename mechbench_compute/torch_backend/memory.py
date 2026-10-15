from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


def read_smi_others(own: int) -> tuple[int | None, str | None]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"nvidia-smi did not answer ({type(e).__name__})"
    total = 0
    rows = answered = 0
    for line in out.splitlines():
        pid, _, used = line.partition(",")
        if not line.strip():
            continue
        rows += 1
        try:
            mib = int(float(used))
            answered += 1
            if int(pid) != own:
                total += mib * 2**20
        except ValueError:
            continue
    if rows and not answered:
        return None, "nvidia-smi reports every process's GPU memory as N/A"
    return total, None


# external: NVML — a process whose memory the driver cannot attribute reports usedGpuMemory as None (pynvml) or NVML_VALUE_NOT_AVAILABLE, 2**64 - 1
def read_nvml_others(own: int, index: int) -> tuple[int | None, str | None]:
    try:
        import pynvml
    except ImportError:
        return None, "pynvml is not installed"
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        procs = []
        for name in ("nvmlDeviceGetComputeRunningProcesses",
                     "nvmlDeviceGetGraphicsRunningProcesses"):
            fn = getattr(pynvml, name, None)
            if fn is not None:
                procs.extend(fn(handle) or ())
    except Exception as e:  # noqa: BLE001
        return None, f"NVML did not answer ({e})"
    used_by: dict[int, int] = {}
    for proc in procs:
        used = getattr(proc, "usedGpuMemory", None)
        if used is None or used >= 2**64 - 1:
            continue
        pid = int(proc.pid)
        used_by[pid] = max(used_by.get(pid, 0), int(used))
    if procs and not used_by:
        return None, "NVML reports every process's GPU memory as not available"
    return sum(v for pid, v in used_by.items() if pid != own), None


def read_others(own: int | None = None, index: int = 0) -> dict[str, Any]:
    own = os.getpid() if own is None else own
    reasons = []
    for source, read in (("nvidia-smi", lambda: read_smi_others(own)),
                         ("nvml", lambda: read_nvml_others(own, index))):
        got, reason = read()
        if got is not None:
            return {"bytes": got, "source": source, "reason": None}
        reasons.append(reason)
    return {"bytes": None, "source": None,
            "reason": "; ".join(reasons) + "; torch.cuda.mem_get_info reads the device as a "
                      "whole, not other processes, so they are counted as nothing"}


def read_others_on_gpu(own: int | None = None, index: int = 0) -> int:
    return read_others(own, index)["bytes"] or 0


# external: an integrated GPU (GB10) on Enverge's VMs — its memory is the machine's, the VM's /proc/meminfo never sees GPU allocations, and cudaMemGetInfo counts the host's reclaimable disk cache as used (59 GB "free" while 80 GB allocated cleanly), so the budget is the device's total less the VM's own use and other processes' GPU memory
def read_integrated_breakdown(index: int, meminfo: Path | None = None) -> dict[str, Any] | None:
    import torch

    host = read_host_used(MEMINFO if meminfo is None else meminfo)
    if host is None:
        return None
    total = torch.cuda.get_device_properties(index).total_memory
    others = read_others(index=index)
    budget = max(0, total - host - (others["bytes"] or 0))
    return {"total": total, "host_used": host, "others": others["bytes"],
            "others_source": others["source"], "others_reason": others["reason"],
            "budget": budget}


def read_integrated_budget(index: int, meminfo: Path | None = None) -> int | None:
    breakdown = read_integrated_breakdown(index, meminfo)
    return None if breakdown is None else breakdown["budget"]


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
    breakdown = read_integrated_breakdown(index)
    if breakdown is None:
        free, _ = torch.cuda.mem_get_info(index)
        budget = free + torch.cuda.memory_reserved(index)
        breakdown = {"total": total, "driver_free": free, "budget": budget,
                     "source": "torch.cuda.mem_get_info"}
    budget = breakdown["budget"]
    fraction = min(1.0, max(0.05, (budget - HEADROOM) / total))
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    logger.info("integrated GPU %d capped at %.3f of %d bytes: %s", index, fraction, total,
                breakdown)
    return fraction
