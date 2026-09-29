from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

EVENT = "event"
STATE = "state"


def run_step(executor: Any, *, graph: Mapping[str, Any], params: Mapping[str, Any],
             outputs: Sequence[Mapping[str, Any]], event: Mapping[str, Any], state: Any,
             inputs: Mapping[str, Any] | None = None,
             secrets: Mapping[str, Any] | None = None,
             on_token: Callable[..., None] | None = None) -> dict[str, Any]:
    if not any(o.get("name") == STATE for o in outputs):
        raise ValueError("a handler declares a `state` output: the state the next event starts from")
    if not isinstance(event, Mapping) or not isinstance(event.get("type"), str):
        raise ValueError("an event is a record with a `type`")
    got = executor.run_sub(graph, {**dict(inputs or {}), EVENT: [dict(event)], STATE: state},
                           params, secrets=secrets, on_token=on_token, outputs=outputs)
    return {"outputs": got, "state": got[STATE]}
