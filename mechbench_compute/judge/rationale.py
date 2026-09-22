from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _rationale(payload: Mapping[str, Any], text: str) -> str:
    value = payload.get("rationale") or payload.get("reason") or ""
    return str(value) if value else text.strip()[:400]
