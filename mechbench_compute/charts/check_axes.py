from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AXIS_SCALES = ("linear", "log", "signed-log")


def check_axes(axes: Any) -> dict[str, Any]:
    if not isinstance(axes, Mapping):
        raise ValueError("records/plot axes is an object: {layer, x, y}")
    out = dict(axes)
    if out.get("layer") is not None:
        out["layer"] = _check_layer(out["layer"])
    for name in ("x", "y"):
        if out.get(name) is not None:
            out[name] = _check_axis(name, out[name])
    return out


def _check_layer(layer: Any) -> dict[str, Any]:
    if not isinstance(layer, Mapping) or "n" not in layer:
        raise ValueError("records/plot axes.layer needs `n`, the number of layers")
    n = int(layer["n"])
    checked: dict[str, Any] = {"n": n}
    if layer.get("global") is not None:
        globals_ = [int(i) for i in layer["global"]]
        bad = [i for i in globals_ if not 0 <= i < n]
        if bad:
            raise ValueError(f"records/plot axes.layer.global names layers outside 0..{n - 1}: {bad}")
        checked["global"] = globals_
    if layer.get("kv_shared_from") is not None:
        k = int(layer["kv_shared_from"])
        if not 0 <= k <= n:
            raise ValueError(f"records/plot axes.layer.kv_shared_from is outside 0..{n}: {k}")
        checked["kv_shared_from"] = k
    return checked


def _check_axis(name: str, axis: Any) -> dict[str, Any]:
    at = f"records/plot axes.{name}"
    if not isinstance(axis, Mapping):
        raise ValueError(f"{at} is an object: {{scale, constant, quantiles, zero}}")
    unknown = sorted(set(axis) - {"scale", "constant", "quantiles", "zero"})
    if unknown:
        raise ValueError(f"{at} names {unknown}; it reads scale, constant, quantiles and zero")
    out: dict[str, Any] = {}
    if axis.get("scale") is not None:
        if axis["scale"] not in AXIS_SCALES:
            raise ValueError(f"{at}.scale must be one of {', '.join(AXIS_SCALES)}, not {axis['scale']!r}")
        out["scale"] = axis["scale"]
    if axis.get("constant") is not None:
        c = axis["constant"]
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not c > 0:
            raise ValueError(f"{at}.constant is the signed-log axis's c, a number above zero, not {c!r}")
        out["constant"] = float(c)
    if axis.get("quantiles") is not None:
        qs = axis["quantiles"]
        if not isinstance(qs, (list, tuple)):
            raise ValueError(f"{at}.quantiles is a list of levels (0.99) or {{q, value}}")
        out["quantiles"] = [_check_quantile(at, i, q) for i, q in enumerate(qs)]
    if axis.get("zero") is not None:
        if not isinstance(axis["zero"], bool):
            raise ValueError(f"{at}.zero is true or false, not {axis['zero']!r}")
        out["zero"] = axis["zero"]
    return out


def _check_quantile(at: str, i: int, q: Any) -> Any:
    level = q.get("q") if isinstance(q, Mapping) else q
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0 <= level <= 1:
        raise ValueError(f"{at}.quantiles[{i}] is a level in 0..1 or a {{q, value}} with one, not {q!r}")
    if not isinstance(q, Mapping):
        return float(level)
    value = q.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{at}.quantiles[{i}].value is the number at that level, not {value!r}")
    return {"q": float(level), "value": float(value)}
