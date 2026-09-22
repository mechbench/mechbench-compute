from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def read_group_key(record: Mapping[str, Any], by: Sequence[str]) -> tuple:
    """The grouping key, read from the record's coordinates and then from
    the record itself.

    A coordinate is where a condition belongs, and most ops put it
    there. Some write the varying thing at the top level instead —
    `intervene/ablate-layers` emits `{id, layer, delta_logp}`, the layer
    being exactly the condition — and grouping by `layer` then silently
    produced ONE row keyed `None` instead of forty-two. The VALUE was
    already read from the top level, so the asymmetry was the bug: a
    field is a field wherever the record carries it (task 000590).
    """
    coords = record.get("coords") or {}
    return tuple(coords.get(k, record.get(k)) for k in by)
