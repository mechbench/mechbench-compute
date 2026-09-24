from __future__ import annotations

from typing import Iterator

import mlx.core as mx

from .errors import CacheKeyError


class ActivationCache:
    def __init__(self, data: dict[str, mx.array] | None = None, *, offset: int = 0):
        self._data: dict[str, mx.array] = dict(data or {})
        self.offset: int = int(offset)

    def __getitem__(self, key: str) -> mx.array:
        if key not in self._data:
            raise CacheKeyError(key, self._data.keys())
        return self._data[key]

    def __setitem__(self, key: str, value: mx.array) -> None:
        self._data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def to_float32(self) -> "ActivationCache":
        return ActivationCache(
            {k: v.astype(mx.float32) for k, v in self._data.items()}
        )

    def __repr__(self) -> str:
        n = len(self._data)
        if n == 0:
            return "ActivationCache(empty)"
        sample = list(self._data.keys())[:3]
        more = f", ... ({n - 3} more)" if n > 3 else ""
        return f"ActivationCache({n} keys: {sample}{more})"


def kv_offset(kv_cache) -> int:
    if not kv_cache:
        return 0
    first = next((c for c in kv_cache if c is not None), None)
    return int(getattr(first, "offset", 0) or 0)
