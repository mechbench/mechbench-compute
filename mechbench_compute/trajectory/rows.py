from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _rows(traj: Mapping[str, Any]) -> list[Any]:
    from mechbench_compute.lexicon import kinds as K

    return K.items_of(traj)
