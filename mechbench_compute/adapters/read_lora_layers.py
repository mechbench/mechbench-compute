from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FORM = ('`lora.layers` is "all", a list of layer indices, or {"from": i, "to": j} '
        '(half-open, either end optional; negative indices count from the end)')


def read_lora_layers(spec: Any, n_layers: int) -> list[int]:
    if spec is None or spec == "all":
        return list(range(n_layers))
    if isinstance(spec, Mapping):
        return read_span(spec, n_layers)
    if not isinstance(spec, (list, tuple)) or not all(is_index(i) for i in spec):
        raise ValueError(f"adapter/train: {FORM}; got {spec!r}")
    if not spec:
        raise ValueError(f"adapter/train: `lora.layers` is empty; {describe_model(n_layers)}")
    beyond = [i for i in spec if not -n_layers <= i < n_layers]
    if beyond:
        raise ValueError(f"adapter/train: `lora.layers` names {', '.join(map(str, beyond))}, "
                         f"and {describe_model(n_layers)}")
    layers = [i % n_layers for i in spec]
    twice = sorted({i for i in layers if layers.count(i) > 1})
    if twice:
        raise ValueError(f"adapter/train: `lora.layers` names layer{'s' if len(twice) > 1 else ''} "
                         f"{', '.join(map(str, twice))} more than once; {describe_model(n_layers)}")
    return sorted(layers)


def read_span(spec: Mapping[str, Any], n_layers: int) -> list[int]:
    if not set(spec) <= {"from", "to"} or not all(is_index(v) for v in spec.values()):
        raise ValueError(f"adapter/train: {FORM}; got {dict(spec)!r}")
    start, stop = spec.get("from", 0), spec.get("to", n_layers)
    if not -n_layers <= start < n_layers:
        raise ValueError(f"adapter/train: `lora.layers` starts at {start}, "
                         f"and {describe_model(n_layers)}")
    if not -n_layers <= stop <= n_layers:
        raise ValueError(f"adapter/train: `lora.layers` ends at {stop}, "
                         f"and {describe_model(n_layers)}")
    layers = list(range(n_layers))[start:stop]
    if not layers:
        raise ValueError(f"adapter/train: `lora.layers` {dict(spec)!r} selects no layer; "
                         f"{describe_model(n_layers)}")
    return layers


def is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def describe_model(n_layers: int) -> str:
    return (f"this model has {n_layers} layers, 0 through {n_layers - 1} "
            f"(-{n_layers} through -1 from the end)")
