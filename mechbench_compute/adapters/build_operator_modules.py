from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx

from mechbench_compute.adapters.operator_module import OperatorModule


def build_operator_modules(spec: Mapping[str, Any], d: int, *, seed: int = 0,
                           values: Mapping[str, Any] | None = None) -> dict[int, OperatorModule]:
    keys = mx.random.split(mx.random.key(int(seed)), len(spec["layers"]))
    modules: dict[int, OperatorModule] = {}
    for layer, key in zip(spec["layers"], keys):
        module = OperatorModule(spec, d, key)
        if values is not None:
            if str(layer) not in values:
                raise ValueError(f"the operator records no parameters for layer {layer}")
            module.load_values(values[str(layer)])
        modules[int(layer)] = module
    return modules
