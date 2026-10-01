from __future__ import annotations

from collections.abc import Sequence


def read_exclude(exclude: Sequence[int] | None, d: int) -> list[int]:
    if exclude is None:
        return []
    if isinstance(exclude, (int, str)) or not isinstance(exclude, Sequence):
        raise TypeError(f"exclude is a list of dimension indices, not {exclude!r}")
    out = sorted({int(i) for i in exclude})
    bad = [i for i in out if not 0 <= i < d]
    if bad:
        raise ValueError(f"exclude names dimensions {bad} outside 0..{d - 1}")
    return out
