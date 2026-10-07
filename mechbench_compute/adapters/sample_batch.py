from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def read_draws(groups: Mapping[str, list[Any]], batch_sizes: Mapping[str, int],
               factories: Mapping[str, Callable[[np.random.Generator], list[Any]]] | None,
               ) -> tuple[list[tuple[str, list[Any], int]], list[tuple[str, Callable, int]]]:
    active = [(g, items, int(batch_sizes.get(g, 0)))
              for g, items in groups.items()
              if items and int(batch_sizes.get(g, 0)) > 0]
    active_factories = [(g, f, int(batch_sizes.get(g, 0)))
                        for g, f in (factories or {}).items()
                        if int(batch_sizes.get(g, 0)) > 0]
    if not active and not active_factories:
        raise ValueError("no non-empty training groups with batch size > 0")
    return active, active_factories


def sample_batch(rng: np.random.Generator, active: list[tuple[str, list[Any], int]],
               active_factories: list[tuple[str, Callable, int]]) -> list[Any]:
    batch: list[Any] = []
    for _, items, k in active:
        take = min(k, len(items))
        for i in rng.choice(len(items), take, replace=False):
            batch.append(items[int(i)])
    for _, factory, k in active_factories:
        for _ in range(k):
            batch.extend(factory(rng))
    return batch
