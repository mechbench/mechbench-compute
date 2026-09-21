from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.interp.target_token_id import _target_token_id


def _tracked_ids(model, record: Mapping[str, Any], *,
                 tracked: Mapping[str, Any] | None = None) -> dict[str, int]:
    """The tokens a read reports on, by name: `tracked` (name → token
    string), the record's own field taking precedence over the op's
    param, in declaration order — the first entry is the target a
    sweep's delta is taken on. A record written before the spellings
    were one may still carry `target`, `outcomes`, `tracks` or `track`;
    those are read after `tracked`, each token under its own text."""
    out: dict[str, int] = {}

    def put(name: Any, text: Any) -> None:
        if str(name) not in out:
            out[str(name)] = _target_token_id(model, str(text))

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
