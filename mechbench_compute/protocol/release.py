from __future__ import annotations

import gc
import sys
import traceback


# external: CPython — an exception and the frames of its traceback form a cycle that only the collector frees, so a stored failure keeps every local of the failed node alive (a 51 GB model on the DGX Spark, 2026-10-12)
def release_frames(exc: BaseException) -> BaseException:
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        if link.__traceback__ is not None:
            traceback.clear_frames(link.__traceback__)
        link = link.__cause__ or link.__context__
    return exc


def release_device_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
