from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.providers import messages as pm
from mechbench_compute.tools.coerce_text import coerce_text
from mechbench_compute.tools.tool_def import ToolDef
from mechbench_compute.tools.tool_run import ToolRun


class Toolbox:
    """The tools one participant may call, and the machinery to run
    them. `block_runner(ref, inputs, params)` is injected by the
    executor for handlers it must own (model blocks, sub-protocols);
    pure blocks run here."""

    def __init__(self, tools: Sequence[Any] = (), *, block_runner=None,
                 session=None) -> None:
        self.tools = [ToolDef.parse(t) for t in tools]
        self._by_name = {t.name: t for t in self.tools}
        if len(self._by_name) != len(self.tools):
            raise ValueError("tool names must be unique within a toolbox")
        self._runner = block_runner
        # A sandbox session (task 000360): stateful, so it binds to the
        # toolbox rather than riding in a handler dict, which is copied
        # into every provenance record and must stay serializable.
        self._session = session
        self.runs: list[ToolRun] = []

    def __bool__(self) -> bool:
        return bool(self.tools)

    def specs(self) -> tuple[pm.ToolSpec, ...]:
        return tuple(t.to_spec() for t in self.tools)

    def call(self, call: pm.ToolCallPart) -> pm.ToolResultPart:
        """Run one tool call. Every failure becomes a result the model
        can read: an unknown name, a handler that raised, a handler
        that was never wired."""
        started = time.monotonic()
        tool = self._by_name.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._by_name)) or "(none)"
            return self._error(call, f"no such tool: {call.name!r}. Available: {known}",
                               {}, started)
        try:
            output = self._dispatch(tool, dict(call.arguments))
        except Exception as e:  # noqa: BLE001 — a tool's failure is data
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
            # A sandbox tool advances the session's snapshot chain; the
            # session is bound to the toolbox, and the arguments are the
            # method's keyword parameters (validated by the schema the
            # model was given, forgiving here).
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
        # A handler block sees the call's arguments on its own port AND
        # as a single record, so an ordinary record block works as a
        # tool without knowing it is one.
        inputs = {"arguments": dict(arguments), "records": [dict(arguments)]}
        if ref is None:
            if self._runner is None:
                raise ValueError(
                    f"tool {tool.name!r} names a protocol handler, which only "
                    "the executor can run")
            return self._runner(handler["protocol"], inputs, params)
        from mechbench_compute.blocks import PURE_BLOCKS

        if ref in PURE_BLOCKS:
            return PURE_BLOCKS[ref](inputs, params)
        # A model block or a sub-protocol: the executor owns those, and
        # a toolbox built without one can say so precisely.
        if self._runner is None:
            raise ValueError(
                f"tool {tool.name!r}: {ref!r} is not a pure block, so it needs "
                "the executor's runner — offer this tool from a protocol node")
        return self._runner(ref, inputs, params)
