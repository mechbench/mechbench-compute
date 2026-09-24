from __future__ import annotations

from typing import Any


def read_records(value: Any) -> list[dict[str, Any]]:
    from mechbench_compute.lexicon import kinds as K

    return K.items_of(value or [])
