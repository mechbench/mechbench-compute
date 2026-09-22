from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mechbench_compute.providers import messages as pm


@dataclass(frozen=True)
class ToolDef:
    """A capability offered to a model. `handler` is
    `{"block": ref, "params": {...}}` or `{"protocol": "<id>"}`; the
    second runs as a nested pipeline through the executor's runner."""

    name: str
    description: str = ""
    schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}})
    handler: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse(value: Any) -> ToolDef:
        if isinstance(value, ToolDef):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "a tool is an object {name, description, schema, handler}, "
                f"not {type(value).__name__}")
        name = str(value.get("name") or "")
        if not name:
            raise ValueError("a tool needs a name — it is what the model calls")
        schema = value.get("schema") or value.get("input_schema") or {
            "type": "object", "properties": {}}
        handler = value.get("handler") or {}
        if handler and not ({"block", "protocol", "sandbox"} & set(handler)):
            raise ValueError(
                f"tool {name!r}: a handler is {{'block': <ref>}}, "
                "{'protocol': <id>} or {'sandbox': <method>}")
        return ToolDef(name=name, description=str(value.get("description", "")),
                       schema=dict(schema), handler=dict(handler))

    def to_spec(self) -> pm.ToolSpec:
        return pm.ToolSpec(name=self.name, description=self.description,
                           input_schema=dict(self.schema))

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "schema": dict(self.schema), "handler": dict(self.handler)}
