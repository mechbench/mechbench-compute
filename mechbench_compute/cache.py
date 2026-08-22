"""ActivationCache: dict-like store of tensors captured during a forward pass.

Values stay in bf16 (the model's native dtype). Cast to float32 right at
the analysis boundary via cache.to_float32() — converting bf16 mx.array
straight to numpy crashes with a PEP 3118 buffer-format error.

The forward pass populates the cache with un-evaluated mx graph nodes, then
calls mx.eval() on the whole batch in one shot. By the time the cache is
returned to the user, every tensor is materialized.
"""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx

from .errors import CacheKeyError


class ActivationCache:
    """A dict-like collection of tensors keyed by hook-point name.

    Standard dict access:
        weights = cache['blocks.23.attn.weights']
        if 'blocks.14.mlp_out' in cache: ...
        for name, tensor in cache.items(): ...

    Missing keys raise CacheKeyError with helpful suggestions, not the bare
    KeyError you'd get from a vanilla dict.
    """

    def __init__(self, data: dict[str, mx.array] | None = None):
        self._data: dict[str, mx.array] = dict(data or {})

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
        """Return a new cache with every tensor cast to float32.

        Use this right before going to numpy. MLX -> numpy on bf16 fails;
        a float32 cast is the standard escape hatch.
        """
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
