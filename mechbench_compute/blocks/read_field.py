from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_field(record: Mapping[str, Any], name: str) -> Any:
    """A record's field by name: a coordinate, else a top-level field, else
    a dot path from the record's root (`metadata.call.usage.output_tokens`,
    `coords.prompt`), the same paths the API's item projection reads. A
    field the record does not have reads `None`.

    A name that is a coordinate or a top-level key is read as that key,
    dots and all, before it is read as a path, so a key that happens to
    hold a dot still reads."""
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
