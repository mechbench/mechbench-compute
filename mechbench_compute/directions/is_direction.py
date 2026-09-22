from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.directions.constants import KIND


def is_direction(d: Any) -> bool:
    """A direction record, under its kind or any alias of it."""
    from mechbench_compute.lexicon import kinds as K

    if not isinstance(d, Mapping) or not isinstance(d.get("kind"), str):
        return False
    try:
        return K.resolve_kind(d["kind"])[0] == KIND
    except KeyError:
        return False
