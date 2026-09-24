from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_field(record: Mapping[str, Any], name: str) -> Any:
    coords = record.get("coords")
    if isinstance(coords, Mapping) and name in coords:
        return coords[name]
    if name in record:
        return record[name]
    if "." not in name:
        return None
    value: Any = record
    for part in name.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value
