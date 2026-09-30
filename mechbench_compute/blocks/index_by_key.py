from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.read_record_key import read_record_key


def index_by_key(items: Sequence[Mapping[str, Any]], key: list[str], side: str,
                 op: str = "diff") -> dict[tuple, Mapping[str, Any]]:
    out: dict[tuple, Mapping[str, Any]] = {}
    for it in items:
        k = tuple(read_record_key(it, n) for n in key)
        if any(v is None for v in k):
            missing = [n for n, v in zip(key, k) if v is None]
            raise ValueError(f"{op}: record {it.get('id')!r} on side {side} has no "
                             f"{', '.join(missing)} to key on")
        if k in out:
            raise ValueError(f"{op}: two records on side {side} at "
                             f"{dict(zip(key, k))}; key on more coordinates")
        out[k] = it
    return out
