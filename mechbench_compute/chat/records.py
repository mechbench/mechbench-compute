from __future__ import annotations

from typing import Any


def _records(value: Any) -> list[dict[str, Any]]:
    """The records to run over, however the port delivered them."""
    from mechbench_compute.lexicon import kinds as K

    return K.items_of(value or [])
