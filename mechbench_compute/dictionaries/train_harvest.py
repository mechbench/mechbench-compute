from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points
from mechbench_compute._mlx import mx


def read_point_name(model: Any, point: str, layer: int) -> str:
    declared = tuple(model.architecture.layer_points)
    if point not in declared or points.LAYOUT.get(point, (0, 0))[1] is not None:
        flat = [p for p in declared if points.LAYOUT.get(p, (0, 0))[1] is None]
        raise ValueError(f"dictionary/train: a dictionary reads one vector per position; {point!r} is not "
                         f"one {model.architecture.name} offers. Its points are {', '.join(flat)}")
    n_layers = int(model.arch.n_layers)
    if not 0 <= layer < n_layers:
        raise ValueError(f"dictionary/train: `layer` {layer} is not a layer of the model; it has {n_layers}")
    return f"blocks.{layer}.{point}"


def read_kept(windows: np.ndarray, bos: int | None) -> np.ndarray:
    keep = np.ones(windows.shape, dtype=bool)
    keep[:, 0] = False
    if bos is not None:
        keep &= windows != bos
    return np.flatnonzero(keep.reshape(-1))


def harvest(model: Any, name: str, windows: Sequence[Sequence[int]], bos: int | None) -> Any:
    ids = np.asarray(windows, dtype=np.int32)
    acts = model.run(mx.array(ids), capture=[name], logits=False).cache[name]
    flat = acts.reshape(-1, acts.shape[-1])
    kept = flat[mx.array(read_kept(ids, bos))].astype(mx.float32)
    mx.eval(kept)
    return kept


def fill(model: Any, name: str, stream: Any, *, needed: int, rows: int, bos: int | None,
         unit: int, on_unit: Callable[[], None] | None, on_windows: Callable[[list], None] | None = None) -> Any:
    parts: list[Any] = []
    have = reported = 0
    while have < needed:
        windows = [stream.next_window() for _ in range(rows)]
        if on_windows is not None:
            on_windows(windows)
        part = harvest(model, name, windows, bos)
        parts.append(part)
        have += int(part.shape[0])
        while on_unit is not None and reported < min(have, needed) // unit:
            reported += 1
            on_unit()
    x = mx.concatenate(parts, axis=0)[:needed] if len(parts) > 1 else parts[0][:needed]
    mx.eval(x)
    return x
