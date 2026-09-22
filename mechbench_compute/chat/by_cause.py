from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _by_cause(errors: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in errors:
        cause = str(e.get("cause", "?"))
        out[cause] = out.get(cause, 0) + 1
    return out
