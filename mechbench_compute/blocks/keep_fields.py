from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.set_field import set_field

_ABSENT = object()


def keep_fields(record: Mapping[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": record["id"]} if "id" in record else {}
    for path in paths:
        value: Any = record
        for part in path.split("."):
            value = value.get(part, _ABSENT) if isinstance(value, Mapping) else _ABSENT
            if value is _ABSENT:
                break
        if value is not _ABSENT:
            out = set_field(out, path, value)
    return out
