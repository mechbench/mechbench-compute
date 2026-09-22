from __future__ import annotations

from typing import Any

from mechbench_compute.tools.toolbox import Toolbox

#: Ready-made definitions for the first tools, so a protocol can offer
#: them by name instead of restating a schema.
BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "calc": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression.",
        "schema": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]},
        "handler": {"block": "tools/calc"},
    },
    "bench.lookup": {
        "name": "bench.lookup",
        "description": "Fetch an object from the bench by its path.",
        "schema": {"type": "object",
                   "properties": {"path": {"type": "string"},
                                  "field": {"type": "string"}},
                   "required": ["path"]},
        "handler": {"block": "tools/lookup"},
    },
}


def toolbox_from(value: Any, *, block_runner=None, session=None) -> Toolbox:
    """A toolbox from a params list: tool objects, or the NAME of a
    built-in ("calc"), so the common case is one word. `session` binds
    a sandbox session (task 000360) for any `{"sandbox": …}` handlers."""
    tools: list[Any] = []
    for entry in value or ():
        if isinstance(entry, str):
            if entry not in BUILTIN_TOOLS:
                raise ValueError(
                    f"unknown built-in tool {entry!r} — "
                    f"{', '.join(sorted(BUILTIN_TOOLS))}, or pass a full "
                    "definition")
            tools.append(BUILTIN_TOOLS[entry])
        else:
            tools.append(entry)
    return Toolbox(tools, block_runner=block_runner, session=session)
