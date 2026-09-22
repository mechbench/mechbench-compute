"""A model's depth landmarks, carried onto a result that has none."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _carry_arch(inputs: Mapping[str, Any], result: Any) -> Any:
    """`result` with the model's depth landmarks on it, when the result
    is an object without them. A bare list has no header to carry them
    on and is returned as is.

    They come from an input: a records op downstream of a model op
    passes them along. An op whose model work happens INSIDE it has no
    such input, and carries them itself — see `_block_map`."""
    if not isinstance(result, dict) or "arch" in result:
        return result
    for value in (inputs or {}).values():
        if isinstance(value, Mapping) and isinstance(value.get("arch"), Mapping):
            return {**result, "arch": dict(value["arch"])}
    return result
