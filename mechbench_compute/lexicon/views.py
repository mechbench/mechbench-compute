from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any


def read_registry() -> Any:
    from mechbench_compute.registry import REGISTRY

    return REGISTRY


class TupleView(Sequence[Any]):
    def __init__(self, read: Callable[[], tuple[Any, ...]]) -> None:
        self._read = read

    def __getitem__(self, i: Any) -> Any:
        return self._read()[i]

    def __len__(self) -> int:
        return len(self._read())

    def __iter__(self) -> Iterator[Any]:
        return iter(self._read())

    def __contains__(self, item: object) -> bool:
        return item in self._read()

    def __eq__(self, other: object) -> bool:
        return tuple(self) == (tuple(other) if isinstance(other, Sequence) else other)

    def __hash__(self) -> int:
        return hash(self._read())

    def __repr__(self) -> str:
        return repr(self._read())


class MapView(Mapping[str, Any]):
    def __init__(self, read: Callable[[], Mapping[str, Any]], pick: Callable[[Any], Any]) -> None:
        self._read = read
        self._pick = pick

    def __getitem__(self, key: str) -> Any:
        return self._pick(self._read()[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._read())

    def __len__(self) -> int:
        return len(self._read())

    def __contains__(self, key: object) -> bool:
        return key in self._read()

    def get(self, key: Any, default: Any = None) -> Any:
        hit = self._read().get(key)
        return default if hit is None else self._pick(hit)

    def __repr__(self) -> str:
        return repr(dict(self.items()))


def view_ops() -> TupleView:
    return TupleView(lambda: read_registry().ops())


def view_by_name() -> MapView:
    return MapView(lambda: read_registry().table().ops, lambda r: r.op)


def view_kinds() -> TupleView:
    return TupleView(lambda: read_registry().kinds())


def view_by_kind() -> MapView:
    return MapView(lambda: read_registry().table().kinds, lambda k: k)
