from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.read_field import read_field


def read_group_key(record: Mapping[str, Any], by: Sequence[str]) -> tuple:
    """The grouping key, read from the record's coordinates and then from
    the record itself.

    A coordinate is where a condition belongs, and most ops put it
    there. Some write the varying thing at the top level instead —
    `intervene/ablate-layers` emits `{id, layer, delta_logp}`, the layer
    being exactly the condition. A field is a field wherever the record
    carries it, and the key must fall back the same way the VALUE does,
    or grouping by `layer` collapses every row onto one key of `None`.
    A dotted name is a path from the record's root (`read_field`).
    """
    return tuple(read_field(record, k) for k in by)
