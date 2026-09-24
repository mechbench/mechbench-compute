from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.read_field import read_field


def read_group_key(record: Mapping[str, Any], by: Sequence[str]) -> tuple:
    return tuple(read_field(record, k) for k in by)
