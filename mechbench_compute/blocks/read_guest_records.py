from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def read_guest_records(op: str, values: Sequence[Any]) -> list[dict[str, Any]]:
    items = list(values[0]) if len(values) == 1 and isinstance(values[0], list) else list(values)
    wrong = [type(v).__name__ for v in items if not isinstance(v, dict)]
    if wrong:
        raise ValueError(f"{op}: every result must be a record (an object); {len(wrong)} are not "
                         f"(the first is {wrong[0]})")
    with_id = sum(1 for r in items if "id" in r)
    if with_id == 0:
        return [{"id": str(i), **r} for i, r in enumerate(items)]
    if with_id < len(items):
        raise ValueError(f"{op}: {len(items) - with_id} of {len(items)} records have no id; give every "
                         f"record one, or none and they are numbered")
    return items
