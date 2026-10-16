from __future__ import annotations

from collections.abc import Callable


class Stop(BaseException):
    def __init__(self, node: str, reason: str) -> None:
        super().__init__(f"{node}: {reason}")
        self.node = node
        self.reason = reason

    def __reduce__(self):
        return type(self), (self.node, self.reason)


class StopCheck:
    def __init__(self, stop: Callable[[], bool] | None) -> None:
        self.stop = stop
        self.stopping = False

    def requested(self) -> bool:
        if not self.stopping and self.stop is not None and self.stop():
            self.stopping = True
        return self.stopping

    def at_item(self, node: str, key: str) -> None:
        if self.requested():
            raise Stop(node, f"stopped after item {key!r}, which is spooled")

    def offer(self, node: str, spool: Callable[[str, dict], None] | None) -> Callable[..., None]:
        def on_checkpoint(capture: Callable[[], dict], *, due: bool) -> None:
            stopping = self.requested()
            if spool is not None and (due or stopping):
                state = capture()
                spool(node, state)
                if stopping:
                    raise Stop(node, f"stopped at step {state.get('step')}, its checkpoint spooled")
            elif stopping:
                raise Stop(node, "stopped at a training boundary, with nothing to spool its state to")

        return on_checkpoint
