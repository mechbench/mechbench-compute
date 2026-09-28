from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mechbench_compute.protocol.protocol_spec import ProtocolSpec

EVENT = "event"
STATE = "state"


def run_step(executor: Any, *, graph: Mapping[str, Any], params: Mapping[str, Any],
             outputs: Sequence[Mapping[str, Any]], event: Mapping[str, Any], state: Any,
             secrets: Mapping[str, Any] | None = None,
             on_token: Callable[..., None] | None = None) -> dict[str, Any]:
    from mechbench_compute import dataflow as dataflow_mod

    if not any(o.get("name") == STATE for o in outputs):
        raise ValueError("a handler declares a `state` output: the state the next event starts from")
    if not isinstance(event, Mapping) or not isinstance(event.get("type"), str):
        raise ValueError("an event is a record with a `type`")
    child = type(executor)(
        on_download=executor._on_download,
        on_download_bytes=executor._on_download_bytes,
        limiter=executor._limiter, budget=executor._budget, on_token=on_token)
    child._model, child._model_id = executor._model, executor._model_id
    try:
        out = child.run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": {**graph, "dataflow": dataflow_mod.DATAFLOW},
                   "params": dict(params),
                   "inputs": {EVENT: [dict(event)], STATE: state},
                   "outputs": [dict(o) for o in outputs],
                   "keep": "all"}),
            secrets=secrets)
    finally:
        executor._model, executor._model_id = child._model, child._model_id
    got = dict(out.payload.get("outputs") or {})
    return {"outputs": got, "state": got[STATE]}
