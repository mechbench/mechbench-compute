from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, fields, replace
from typing import Any

from mechbench_compute import snapshots as fs

ENV_PREFIX = "MECHBENCH_SANDBOX_CEILING_"


@dataclass(frozen=True)
class Ceilings:
    memory_mb: int = 4096
    fuel: int = 100_000_000_000
    wall_seconds: float = 900.0
    output_bytes: int = 256 * 1024 * 1024
    max_files: int = fs.MAX_FILES
    max_bytes: int = fs.MAX_BYTES
    disk_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for f in fields(self):
            check_limit(f.name, getattr(self, f.name), what="ceiling")


_LOCK = threading.Lock()
_SET: Ceilings | None = None


def check_limit(name: str, value: Any, *, what: str = "limit") -> None:
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "wall_seconds":
        ok = number and math.isfinite(value) and value > 0
        kind = "a finite number of seconds above 0"
    else:
        ok = number and isinstance(value, int) and value >= 1
        kind = "a whole number, at least 1"
    if not ok:
        raise ValueError(f"sandbox {what} {name} must be {kind}, not {value!r}")


def set_ceilings(ceilings: Ceilings | None) -> None:
    global _SET
    with _LOCK:
        _SET = ceilings


def read_ceilings() -> Ceilings:
    with _LOCK:
        if _SET is not None:
            return _SET
    given: dict[str, Any] = {}
    for f in fields(Ceilings):
        raw = os.environ.get(ENV_PREFIX + f.name.upper())
        if raw is None or not raw.strip():
            continue
        try:
            given[f.name] = float(raw) if f.name == "wall_seconds" else int(raw)
        except ValueError:
            raise ValueError(
                f"{ENV_PREFIX}{f.name.upper()}={raw!r} is not a number") from None
    return replace(Ceilings(), **given) if given else Ceilings()
