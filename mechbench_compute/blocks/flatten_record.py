from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def flatten_record(d: Any, pre: str = "") -> dict[str, Any]:
    if not isinstance(d, Mapping):
        return {}
    out: dict[str, Any] = {}
    for f, v in d.items():
        if isinstance(v, Mapping) and v:
            out.update(flatten_record(v, f"{pre}{f}."))
        else:
            out[f"{pre}{f}"] = v
    return out
