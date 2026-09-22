from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def expand_grid(item: Any) -> list[dict[str, Any]] | None:
    """A grid's cells as rows, or None when the item is not one.

    `intervene/patch`, `intervene/ablate-heads` and the lens read out
    GRIDS — named measures indexed by `axes` — because that is the
    shape a heat map is. A chart takes rows, so the cells become them:
    one row per cell, the axes as fields (with `token` beside
    `position` when the grid carries the tokens), each measure a column.
    """
    if not isinstance(item, Mapping):
        return None
    axes = item.get("axes")
    measures = item.get("measures")
    if not (isinstance(axes, list) and axes and isinstance(measures, Mapping) and measures):
        return None
    first = next(iter(measures.values()))
    shape: list[int] = []
    cursor: Any = first
    for _ in axes:
        if not isinstance(cursor, list):
            return None
        shape.append(len(cursor))
        cursor = cursor[0] if cursor else None
    tokens = item.get("tokens") if isinstance(item.get("tokens"), list) else None
    base = {"id": item.get("id"), **(item.get("coords") or {})}

    def cells(index: list[int]) -> dict[str, Any]:
        row = dict(base)
        for axis, i in zip(axes, index, strict=True):
            row[str(axis)] = i
            if axis == "position" and tokens is not None and i < len(tokens):
                row["token"] = tokens[i]
        for name, values in measures.items():
            v: Any = values
            for i in index:
                v = v[i] if isinstance(v, list) and i < len(v) else None
            row[str(name)] = v
        return row

    out: list[dict[str, Any]] = []
    index = [0] * len(axes)

    def walk(depth: int) -> None:
        if depth == len(axes):
            out.append(cells(index))
            return
        for i in range(shape[depth]):
            index[depth] = i
            walk(depth + 1)

    walk(0)
    return out
