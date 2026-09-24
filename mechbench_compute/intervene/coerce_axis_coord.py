from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def coerce_axis_coord(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return coerce_axis_coord(value[0])
        return "+".join(str(coerce_axis_coord(v)) for v in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)
