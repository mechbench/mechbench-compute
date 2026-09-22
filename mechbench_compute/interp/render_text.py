from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute._mlx import mx
from mechbench_compute.distill import render


def render_text(model, record: Mapping[str, Any], text: str) -> mx.array:
    """One side of a pair, rendered as the pair's record says (raw unless
    it carries `template: "chat"`)."""
    return render(model, {"id": record.get("id"), "text": text,
                          "template": record.get("template")}).array
