from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class Monoid:
    """`partial(records, params)` → P; `identity()` → P; `merge(P, P)` → P
    (associative, commutative); `finalize(P, params)` → the block's
    output. `partial(all) == merge over any partition of all` exactly."""

    def identity(self) -> Any:
        raise NotImplementedError

    def partial(self, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def merge(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def finalize(self, p: Any, params: Mapping[str, Any]) -> Any:
        raise NotImplementedError
