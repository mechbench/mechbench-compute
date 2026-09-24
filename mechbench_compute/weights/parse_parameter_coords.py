from __future__ import annotations

from typing import Any


def parse_parameter_coords(name: str) -> dict[str, Any]:
    parts = name.split(".")
    coords: dict[str, Any] = {}
    if parts[0] == "layers" and len(parts) > 2 and parts[1].isdigit():
        coords["layer"] = int(parts[1])
        rest = parts[2:]
        if len(rest) > 1 and rest[0] in ("self_attn", "mlp"):
            coords["container"] = rest[0]
            coords["projection"] = rest[1]
        coords["module"] = ".".join(parts[:-1]) if len(parts) > 2 else name
    else:
        coords["module"] = ".".join(parts[:-1]) or name
    coords["parameter"] = parts[-1]
    return coords
