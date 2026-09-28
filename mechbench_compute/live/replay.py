from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mechbench_compute.live.run_step import run_step

PARAM = "param"


def replay(executor: Any, *, graph: Mapping[str, Any], params: Mapping[str, Any],
           outputs: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]], state: Any,
           secrets: Mapping[str, Any] | None = None,
           on_step: Callable[[int, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    bound = dict(params)
    steps: list[dict[str, Any]] = []
    for i, event in enumerate(events):
        if event.get("type") == PARAM:
            name = event.get("name")
            if not isinstance(name, str) or name not in bound:
                raise ValueError(f"event {event.get('id')!r} changes {name!r}, which the live run does not bind")
            bound[name] = event.get("value")
            steps.append({"event": dict(event), "params": dict(bound)})
        else:
            got = run_step(executor, graph=graph, params=bound, outputs=outputs,
                           event=event, state=state, secrets=secrets)
            state = got["state"]
            steps.append({"event": dict(event), "outputs": got["outputs"]})
        if on_step is not None:
            on_step(i, steps[-1])
    return {"kind": "run/replay", "steps": steps, "state": state, "params": bound}
