from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.chat.constants import ENDINGS


def count_endings(items: Sequence[Any]) -> dict[str, int]:
    """A node header's `ended`: its items counted by
    `metadata.sampling.ended`, every ending present, zero when none —
    so `max_tokens: 0` is said, not implied."""
    out = dict.fromkeys(ENDINGS, 0)
    for it in items:
        meta = (it.get("metadata") if isinstance(it, Mapping) else None) or {}
        ended = (meta.get("sampling") or {}).get("ended")
        if ended in out:
            out[ended] += 1
    return out
