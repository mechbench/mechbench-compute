from __future__ import annotations

import re


def _pattern(point: str) -> re.Pattern[str]:
    """A parameter point as a matcher. `*` stands for one segment, so
    `layers.*.mlp.down_proj` is every layer's down projection; anything
    else is literal. A point that names a MODULE matches its parameters
    (`…q_proj` matches `…q_proj.weight`)."""
    body = re.escape(point).replace(r"\*", r"[^.]+")
    return re.compile(rf"^{body}(\.[^.]+)?$")
