from __future__ import annotations

from typing import Any

HEADROOM = 8 * 2**30


def read_index(device: Any) -> int:
    import torch

    return device.index if device.index is not None else torch.cuda.current_device()


def read_free_bytes(device: Any) -> int | None:
    if getattr(device, "type", None) != "cuda":
        return None
    import torch

    free, _total = torch.cuda.mem_get_info(read_index(device))
    return int(free)


# external: the caching allocator on an integrated GPU (GB10) — it takes the machine's memory until the VM is reset instead of raising torch.OutOfMemoryError; the VM's /proc/meminfo does not see GPU allocations, cudaMemGetInfo does (Enverge's DGX Spark, 2026-10-12)
def cap_integrated(device: Any) -> float | None:
    if getattr(device, "type", None) != "cuda":
        return None
    import torch

    index = read_index(device)
    props = torch.cuda.get_device_properties(index)
    if not getattr(props, "is_integrated", False):
        return None
    free, total = torch.cuda.mem_get_info(index)
    held = torch.cuda.memory_reserved(index)
    fraction = min(1.0, max(0.05, (free + held - HEADROOM) / total))
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction
