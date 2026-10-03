from __future__ import annotations

import math

PLACES = 3


def say_number(x: float, places: int = PLACES) -> str:
    if isinstance(x, bool) or not math.isfinite(float(x)):
        return str(x)
    if float(x).is_integer():
        return str(int(x))
    scale = 10 ** places
    rounded = math.floor(float(x) * scale + 0.5) / scale
    if rounded == 0:
        return "0"
    if rounded.is_integer():
        return str(int(rounded))
    return repr(rounded)
