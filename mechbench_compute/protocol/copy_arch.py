from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def copy_arch(inputs: Mapping[str, Any], result: Any) -> Any:
    if not isinstance(result, dict) or "arch" in result:
        return result
    for value in (inputs or {}).values():
        if isinstance(value, Mapping) and isinstance(value.get("arch"), Mapping):
            return {**result, "arch": dict(value["arch"])}
    return result
