from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mechbench_compute.spans.node_span import NodeSpan


def find_memory_counter() -> Any:
    try:
        import mlx.core as mx
    except ImportError:
        return None
    for host in (mx, getattr(mx, "metal", None)):
        if host is not None and hasattr(host, "get_peak_memory"):
            return host
    return None


@contextmanager
def open_span() -> Iterator[NodeSpan]:
    parent = NodeSpan.active
    counter = find_memory_counter()
    if counter is not None and parent is None:
        counter.reset_peak_memory()
    span = NodeSpan()
    NodeSpan.active = span
    started = time.perf_counter()
    try:
        yield span
    finally:
        NodeSpan.active = parent
        span.close(time.perf_counter() - started,
                   int(counter.get_peak_memory()) if counter is not None else None)
        if parent is not None:
            parent.add(model_load_seconds=span.model_load_seconds,
                       download_seconds=span.download_seconds, **span.counts)
