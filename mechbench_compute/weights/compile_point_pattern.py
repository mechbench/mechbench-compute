from __future__ import annotations

import re


def compile_point_pattern(point: str) -> re.Pattern[str]:
    body = re.escape(point).replace(r"\*", r"[^.]+")
    return re.compile(rf"^{body}(\.[^.]+)?$")
