from __future__ import annotations

from typing import Any


def resolve_module(lm: Any, name: str) -> tuple[Any, str]:
    """`(module, attribute)` for a parameter name, so the tensor can be
    written back where it came from."""
    parts = name.split(".")
    node = getattr(lm, "model", lm)
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node, parts[-1]
