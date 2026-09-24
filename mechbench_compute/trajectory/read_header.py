from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_header(traj: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in traj.items()
            if k not in ("kind", "item_kind", "key", "items", "rows")}
