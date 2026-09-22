from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _coords_of(record: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """The record's coordinates. A grouping is a coordinate; the retired
    `label` field is read as the `label` coordinate so older records
    group as they did. (A document keeps its coords under `metadata`.)"""
    coords = dict(record.get("coords") or (record.get("metadata") or {}).get("coords") or {})
    label = record.get("label")
    if label is not None and "label" not in coords:
        coords["label"] = label
    return coords
