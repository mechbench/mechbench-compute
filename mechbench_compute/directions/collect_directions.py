from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.directions.is_direction import is_direction


def collect_directions(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    got: list[Mapping[str, Any]] = []
    listed = inputs.get("directions")
    if isinstance(listed, list):
        got.extend(listed)
    for k in sorted(inputs):
        v = inputs[k]
        if is_direction(v) and k != "directions":
            got.append(v)
    if not got:
        raise ValueError("no direction objects on the inputs (ports) or params.directions")
    return got
