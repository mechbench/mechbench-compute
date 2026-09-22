from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import shapes as S


def read_space(d: Mapping[str, Any]) -> dict[str, Any]:
    return S.space_of(d)
