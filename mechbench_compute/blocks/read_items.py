from __future__ import annotations

from typing import Any


def read_items(x: Any) -> list[dict[str, Any]]:
    """The items of an input, however it arrived — the lexicon's one
    reader, for a `collection`, a bare list, or an older plural object."""
    from mechbench_compute.lexicon import kinds as K

    return K.items_of(x)
