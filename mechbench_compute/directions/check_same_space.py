from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import shapes as S


def check_same_space(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    S.same_space(a, b, what="directions")
