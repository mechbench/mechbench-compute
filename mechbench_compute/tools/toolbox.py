from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.providers import messages as pm
from mechbench_compute.tools.coerce_text import coerce_text
from mechbench_compute.tools.tool_def import ToolDef
from mechbench_compute.tools.tool_run import ToolRun


class Toolbox:
    def __init__(self, tools: Sequence[Any] = (), *, block_runner=None,
                 session=None) -> None:
        self.tools = [ToolDef.parse(t) for t in tools]
        self._by_name = {t.name: t for t in self.tools}
        if len(self._by_name) != len(self.tools):
            raise ValueError("tool names must be unique within a toolbox")
        self._runner = block_runner
        self._session = session
        self.runs: list[ToolRun] = []

    def __bool__(self) -> bool:
        return bool(self.tools)

    def specs(self) -> tuple[pm.ToolSpec, ...]:
        return tuple(t.to_spec() for t in self.tools)

    def call(self, call: pm.ToolCallPart) -> pm.ToolResultPart:
        started = time.monotonic()
        tool = self._by_name.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._by_name)) or "(none)"
            return self._error(call, f"no such tool: {call.name!r}. Available: {known}",
                               {}, started)
        try:
            output = self._dispatch(tool, dict(call.arguments))
        except Exception as e:  # noqa: BLE001
            return self._error(call, f"{type(e).__name__}: {e}", tool.handler, started)
        run = ToolRun(tool=tool.name, arguments=dict(call.arguments),
                      handler=dict(tool.handler), output=output,
                      duration_ms=int((time.monotonic() - started) * 1000))
        self.runs.append(run)
        return pm.ToolResultPart(tool_call_id=call.id, content=coerce_text(output))

    def _error(self, call: pm.ToolCallPart, message: str,
               handler: Mapping[str, Any], started: float) -> pm.ToolResultPart:
        self.runs.append(ToolRun(
            tool=call.name, arguments=dict(call.arguments), handler=dict(handler),
            error=message, duration_ms=int((time.monotonic() - started) * 1000)))
        return pm.ToolResultPart(tool_call_id=call.id, content=message,
                                 is_error=True)

    def _dispatch(self, tool: ToolDef, arguments: Mapping[str, Any]) -> Any:
        handler = tool.handler
        if not handler:
            raise ValueError(
                f"tool {tool.name!r} has no handler — it can be offered to a "
                "model but not run")
        method = handler.get("sandbox")
        if method is not None:
            if self._session is None:
                raise ValueError(
                    f"tool {tool.name!r} is a sandbox tool, but this toolbox "
                    "was built without a session — offer it from a node that "
                    "declares a `sandbox` image")
            fn = getattr(self._session, str(method), None)
            if fn is None:
                raise ValueError(
                    f"tool {tool.name!r}: the session has no method "
                    f"{method!r}")
            return fn(**dict(arguments))
        ref = handler.get("block")
        params = dict(handler.get("params") or {})
        inputs = {"arguments": dict(arguments), "records": [dict(arguments)]}
        if ref is None:
            if self._runner is None:
                raise ValueError(
                    f"tool {tool.name!r} names a protocol handler, which only "
                    "the executor can run")
            return self._runner(handler["protocol"], inputs, params)
        from mechbench_compute import ops

        if ref in ops.find_standalone():
            return ops.run_standalone(ref, inputs, params)
        if self._runner is None:
            raise ValueError(
                f"tool {tool.name!r}: {ref!r} is not a pure block, so it needs "
                "the executor's runner — offer this tool from a protocol node")
        return self._runner(ref, inputs, params)
