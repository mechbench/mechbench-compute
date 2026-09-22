from __future__ import annotations

from typing import Any


def _transcripts(value: Any) -> list[dict[str, Any]]:
    from mechbench_compute.lexicon import kinds as K

    items = K.items_of(value or [])
    for t in items:
        if not isinstance(t.get("messages"), list):
            raise ValueError(
                f"transcript {t.get('id')!r} has no `messages`; a text/transcript "
                "carries its messages at the top level")
    return items
