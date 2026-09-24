from __future__ import annotations

from typing import Any

from mechbench_compute.tools.toolbox import Toolbox

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


def build_toolbox(value: Any, *, block_runner=None, session=None) -> Toolbox:
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
