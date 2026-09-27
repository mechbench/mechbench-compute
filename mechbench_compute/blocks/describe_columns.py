from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def describe_columns(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[dict[str, str]]:
    return [{"name": k, "dtype": _infer_dtype(rows, k)} for k in names]


def _infer_dtype(rows: Sequence[Mapping[str, Any]], name: str) -> str:
    vals = [r.get(name) for r in rows if r.get(name) is not None]
    numeric = vals and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
    return "number" if numeric else "string"
