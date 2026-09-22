from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _sandbox_and_tools(params: Mapping[str, Any]) -> tuple[Any, tuple[Any, ...]]:
    """`(image, combined_tool_specs)` for a node. When the node
    declares a `sandbox` image, its tools are appended to any the
    protocol listed, so the model is offered both. `image` is None when
    there is no sandbox — the ordinary tool path, unchanged."""
    tool_specs = list(params.get("tools") or ())
    if params.get("sandbox") is None:
        return None, tuple(tool_specs)
    from mechbench_compute.sandbox_session import SandboxImage, SandboxSession

    image = SandboxImage.parse(params["sandbox"])
    return image, tuple(tool_specs + SandboxSession(image).tool_defs())
