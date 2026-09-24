from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class Monoid:
    def identity(self) -> Any:
        raise NotImplementedError

    def partial(self, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def merge(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def finalize(self, p: Any, params: Mapping[str, Any]) -> Any:
        raise NotImplementedError
