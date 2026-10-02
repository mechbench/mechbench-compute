from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.intervene.operator_refused import OperatorRefused

PORT_WORD = "direction"


class Mask:
    def __init__(self, *, dims: Sequence[int] | None = None, basis: np.ndarray | None = None,
                 invert: bool = False) -> None:
        self.dims = None if dims is None else [int(i) for i in dims]
        self.invert = bool(invert) and dims is not None
        self.basis = basis
        self.reader = None
        self._arrays: dict[Any, Any] = {}
        if basis is not None:
            norms = np.linalg.norm(basis, axis=1, keepdims=True)
            if (norms == 0).any() or np.linalg.matrix_rank(basis / np.maximum(norms, 1e-300),
                                                           tol=1e-6) < len(basis):
                raise OperatorRefused(
                    "MASK_DEGENERATE", f"the mask's {len(basis)} directions span fewer dimensions than "
                    "there are directions, so coordinates in that frame are not one answer: drop the "
                    "dependent ones", construct="mask")
            self.reader = basis.T @ np.linalg.inv(basis @ basis.T)

    def read_width(self, d: int) -> int:
        if self.basis is not None:
            return len(self.basis)
        return d if self.dims is None else len(self.read_indices(d))

    def read_indices(self, d: int) -> list[int]:
        if not self.invert:
            return list(self.dims or ())
        named = set(self.dims or ())
        return [i for i in range(d) if i not in named]

    def check(self, d: int, point: str) -> None:
        if self.dims is not None:
            beyond = [i for i in self.dims if not 0 <= i < d]
            if beyond:
                raise OperatorRefused(
                    "MASK_OUT_OF_RANGE", f"the mask names dimension{'' if len(beyond) == 1 else 's'} "
                    f"{', '.join(str(i) for i in beyond)}, and {point!r} is {d} wide here: dimensions are "
                    f"0 through {d - 1}", construct=str(beyond[0]))
        if self.basis is not None and self.basis.shape[1] != d:
            raise OperatorRefused(
                "MASK_WIDTH_MISMATCH", f"the mask's directions are {self.basis.shape[1]} wide, and "
                f"{point!r} is {d} wide here", construct="mask")

    def read_row(self, row: np.ndarray) -> np.ndarray:
        if self.basis is not None:
            return row.astype(np.float64) @ self.reader
        return row if self.dims is None else row[self.read_indices(len(row))]

    def read(self, a: Any) -> Any:
        if self.basis is not None:
            return a @ self._array("reader", self.reader)
        if self.dims is None:
            return a
        return mx.take(a, self._array(("indices", a.shape[-1]), self.read_indices(a.shape[-1]), mx.int32),
                       axis=-1)

    def write(self, a: Any, x: Any, y: Any, strength: float) -> Any:
        y = mx.broadcast_to(y if isinstance(y, mx.array) else mx.array(y, dtype=mx.float32), x.shape)
        if self.basis is not None:
            step = y - x if strength == 1.0 else strength * (y - x)
            return a + step @ self._array("basis", self.basis)
        new = y if strength == 1.0 else x + strength * (y - x)
        if self.dims is None:
            return new
        idx = self._array(("indices", a.shape[-1]), self.read_indices(a.shape[-1]), mx.int32)
        where = mx.broadcast_to(idx.reshape((1,) * (a.ndim - 1) + (-1,)), (*a.shape[:-1], idx.size))
        return mx.put_along_axis(a, where, new, axis=-1)

    def _array(self, key: Any, value: Any, dtype: Any = None) -> Any:
        if key not in self._arrays:
            self._arrays[key] = mx.array(np.asarray(value, dtype=np.int32 if dtype is mx.int32 else np.float32))
        return self._arrays[key]


def read_mask(value: Any, *, invert: bool = False) -> Mask:
    from mechbench_compute.directions.is_direction import is_direction
    from mechbench_compute.lexicon import kinds as K

    if value is None:
        return Mask()
    if value == PORT_WORD:
        raise OperatorRefused(
            "MASK_INVALID", "`mask: \"direction\"` reads the node's `direction` port, and no direction "
            "arrived there", construct=PORT_WORD)
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if isinstance(value, Sequence) and not isinstance(value, str) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in value):
        dims = list(value)
        if not dims:
            raise OperatorRefused("MASK_INVALID", "an empty mask selects nothing: name a dimension, or "
                                  "leave `mask` out to act on every coordinate", construct="[]")
        twice = sorted({i for i in dims if dims.count(i) > 1})
        if twice:
            raise OperatorRefused("MASK_INVALID", f"the mask names dimension {twice[0]} twice",
                                  construct=str(twice[0]))
        return Mask(dims=dims, invert=invert)
    if isinstance(value, Mapping) and K.item_kind_of(value) is not None:
        directions = list(K.items_of(value))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        directions = list(value)
    else:
        directions = [value]
    if not directions or not all(is_direction(d) for d in directions):
        raise OperatorRefused(
            "MASK_INVALID", "a mask is a list of dimensions, a direction (`direction/vector`), a frame "
            "(several of them, as a list or a collection) or `\"direction\"` for the node's `direction` "
            f"port; not {type(value).__name__}", construct="mask")
    rows = [np.asarray(d["vector"], dtype=np.float64).reshape(-1) for d in directions]
    if len({r.size for r in rows}) > 1:
        raise OperatorRefused("MASK_INVALID", "the frame's directions are "
                              f"{' and '.join(sorted({str(r.size) for r in rows}))} wide: one space, one width",
                              construct="mask")
    return Mask(basis=np.stack(rows))
