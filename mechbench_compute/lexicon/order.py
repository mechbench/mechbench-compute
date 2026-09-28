from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import COLLECTION

_SPACE_ORDER = ("model", "layer", "point", "head", "d")
_DIGITS = re.compile(r"([0-9]+)")


def _sort_value(v: Any) -> tuple[int, Any]:
    if v is None:
        return (0, "")
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, (int, float)):
        return (1, v)
    if isinstance(v, str):
        return (2, _natural(v))
    if isinstance(v, Mapping):
        keys = (_SPACE_ORDER if set(v) == set(_SPACE_ORDER) else tuple(sorted(v)))
        return (3, tuple((k, _sort_value(v[k])) for k in keys))
    if isinstance(v, (list, tuple)):
        return (4, tuple(_sort_value(x) for x in v))
    return (5, json.dumps(v, sort_keys=True, default=str))


def _natural(s: str) -> tuple[Any, ...]:
    parts = _DIGITS.split(s)
    return tuple(p if i % 2 == 0 else (int(p), len(p)) for i, p in enumerate(parts))


def _read_path(item: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(item, Mapping):
            return None
        item = item.get(part)
    return item


def canonical_collection(obj: Any) -> Any:
    if not isinstance(obj, Mapping) or obj.get("kind") != COLLECTION:
        return obj
    key = list(obj.get("key") or [])
    order = [str(f) for f in obj.get("order_by") or []]
    fields = order + [k for k in key if k not in order]
    items = list(obj.get("items") or [])
    if fields:
        items = sorted(items, key=lambda it: tuple(_sort_value(_read_path(it, f)) for f in fields))
    out = dict(obj)
    out["items"] = items
    return out
