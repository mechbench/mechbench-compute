from __future__ import annotations

import threading
from typing import Any, ClassVar

COUNTS = ("tokens_in", "tokens_out", "forwards", "backwards", "bytes_captured")

SPAN_FIELDS = ("peak_memory_bytes", *COUNTS, "model_load_seconds", "compute_seconds",
               "ambient", "quiet")


class NodeSpan:
    active: ClassVar[NodeSpan | None] = None

    def __init__(self) -> None:
        self.counts = dict.fromkeys(COUNTS, 0)
        self.model_load_seconds: float | None = None
        self.download_seconds = 0.0
        self.peak_memory_bytes: int | None = None
        self.compute_seconds: float | None = None
        self.lock = threading.Lock()

    def add(self, *, model_load_seconds: float | None = None,
            download_seconds: float | None = None, **counts: int) -> None:
        with self.lock:
            for name, n in counts.items():
                self.counts[name] += int(n)
            if model_load_seconds is not None:
                self.model_load_seconds = (self.model_load_seconds or 0.0) + model_load_seconds
            if download_seconds is not None:
                self.download_seconds += download_seconds

    def close(self, wall_seconds: float, peak_memory_bytes: int | None) -> None:
        self.peak_memory_bytes = peak_memory_bytes
        self.compute_seconds = round(max(
            0.0, wall_seconds - (self.model_load_seconds or 0.0) - self.download_seconds), 6)
        if self.model_load_seconds is not None:
            self.model_load_seconds = round(self.model_load_seconds, 6)

    def to_dict(self) -> dict[str, Any]:
        return {"peak_memory_bytes": self.peak_memory_bytes, **self.counts,
                "model_load_seconds": self.model_load_seconds,
                "compute_seconds": self.compute_seconds,
                "ambient": None, "quiet": None}
