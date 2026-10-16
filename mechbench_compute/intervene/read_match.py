from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.intervene.spec_error import SpecError


def read_match(row: Any, op: str, source: Any) -> str | None:
    if not isinstance(row, Mapping):
        return None
    match = row.get("match")
    if match is None:
        return None
    if not isinstance(match, str) or not match:
        raise SpecError(f"`row.match` names a field of the records, such as \"id\"; not {match!r}")
    if "index" in row:
        raise SpecError("`row` picks a source row by `index` or by `match`, not both")
    if op != "patch":
        raise SpecError(f"`row` is `patch`'s; this item's op is {op!r}")
    if source is None:
        raise SpecError("`row.match` pairs each record with a `source` row; the item has no "
                        "source and none arrived on the node's `source` port")
    return match
