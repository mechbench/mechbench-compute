from __future__ import annotations

from collections.abc import Callable


def due_only(spool: Callable[[dict], object]) -> Callable[..., None]:
    def on_checkpoint(capture: Callable[[], dict], *, due: bool) -> None:
        if due:
            spool(capture())

    return on_checkpoint
