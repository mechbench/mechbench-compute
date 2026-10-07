from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.intervene.operator_refused import OperatorRefused
from mechbench_compute.intervene.read_mask import Mask

REDUCES = ("mean",)


def check_constants(constants: Any, names: tuple[str, ...]) -> None:
    if not isinstance(constants, Mapping):
        raise OperatorRefused("CONSTANT_INVALID", "`constants` is an object of named values, "
                              "`{\"k\": 2.0}`", construct="constants")
    for name in names:
        if name not in constants:
            raise OperatorRefused(
                "OPERATOR_NAME_UNBOUND", f"`f` reads `{name}`, which is neither `x` nor one of the "
                f"item's constants ({', '.join(sorted(constants)) or 'none given'})", construct=name)
    for name, value in constants.items():
        if name == "x":
            raise OperatorRefused("CONSTANT_INVALID", "`x` is the masked coordinates; give the constant "
                                  "another name", construct=name)
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        numbers = isinstance(value, list) and bool(value) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        source = (isinstance(value, Mapping) and set(value) == {"source"}
                  and value["source"] in REDUCES)
        if not (number or numbers or source):
            raise OperatorRefused(
                "CONSTANT_INVALID", f"constant `{name}` is a number, a list of numbers (one per "
                "coordinate of `x`) or `{\"source\": \"mean\"}` (the mean of the `source` rows, read in "
                f"the mask's coordinates); not {value!r}"[:400], construct=name)


def bind_constants(constants: Mapping[str, Any], mask: Mask, d: int, point: str,
                   rows: np.ndarray | None, xp: Any) -> dict[str, Any]:
    width = mask.read_width(d)
    bound: dict[str, Any] = {}
    for name, value in constants.items():
        if isinstance(value, Mapping):
            if rows is None:
                raise OperatorRefused(
                    "CONSTANT_INVALID", f"constant `{name}` binds from `source`, and the item has no "
                    "source and none arrived on the node's `source` port", construct=name)
            row = rows.mean(0)
            if row.size != d:
                raise OperatorRefused(
                    "CONSTANT_INVALID", f"constant `{name}` binds from `source`, whose rows are "
                    f"{row.size} wide, and {point!r} is {d} wide here", construct=name)
            bound[name] = xp.array(np.asarray(mask.read_row(row), dtype=np.float32))
        elif isinstance(value, (int, float)):
            bound[name] = float(value)
        else:
            if len(value) != width:
                raise OperatorRefused(
                    "CONSTANT_INVALID", f"constant `{name}` holds {len(value)} values, one per "
                    f"coordinate, and `x` has {width} here", construct=name)
            bound[name] = xp.array(np.asarray(value, dtype=np.float32))
    return bound
