from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_record_key(record: Mapping[str, Any], name: str) -> Any:
    if name == "id":
        return record.get("id")
    for where in (record.get("coords"), record, (record.get("metadata") or {}).get("coords")):
        if isinstance(where, Mapping) and name in where:
            return where[name]
    return None
