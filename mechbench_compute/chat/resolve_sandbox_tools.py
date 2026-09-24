from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_sandbox_tools(params: Mapping[str, Any]) -> tuple[Any, tuple[Any, ...]]:
    tool_specs = list(params.get("tools") or ())
    if params.get("sandbox") is None:
        return None, tuple(tool_specs)
    from mechbench_compute.sandbox_session import SandboxImage, SandboxSession

    image = SandboxImage.parse(params["sandbox"])
    return image, tuple(tool_specs + SandboxSession(image).tool_defs())
