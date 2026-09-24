from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_record_coords(record: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    coords = dict(record.get("coords") or (record.get("metadata") or {}).get("coords") or {})
    label = record.get("label")
    if label is not None and "label" not in coords:
        coords["label"] = label
    return coords
