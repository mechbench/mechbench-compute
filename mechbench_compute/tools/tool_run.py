from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolRun:
    tool: str
    arguments: Mapping[str, Any]
    handler: Mapping[str, Any]
    output: Any = None
    error: str = ""
    duration_ms: int = 0

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tool": self.tool,
                               "arguments": dict(self.arguments),
                               "handler": dict(self.handler),
                               "duration_ms": self.duration_ms}
        if self.error:
            out["error"] = self.error
        return out
