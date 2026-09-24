from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.interp.encode_target_token import encode_target_token


def collect_tracked_ids(model, record: Mapping[str, Any], *,
                        tracked: Mapping[str, Any] | None = None) -> dict[str, int]:
    out: dict[str, int] = {}

    def put(name: Any, text: Any) -> None:
        if str(name) not in out:
            out[str(name)] = encode_target_token(model, str(text))

    for name, text in dict(record.get("tracked") or tracked or {}).items():
        put(name, text)
    if record.get("target"):
        put(record["target"], record["target"])
    for o in list(record.get("outcomes") or []):
        put(o, o)
    for name, text in dict(record.get("tracks") or {}).items():
        put(name, text)
    if record.get("track"):
        put(record["track"], record["track"])
    return out
