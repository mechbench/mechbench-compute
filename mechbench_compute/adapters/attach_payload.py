from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.adapters.attach_operators import OperatorHandle, attach_operators
from mechbench_compute.adapters.build_operator_modules import build_operator_modules
from mechbench_compute.adapters.read_operator_spec import FIELDS, read_operator_spec


def attach_payload(lm, payload: Mapping[str, Any]) -> OperatorHandle:
    record = payload.get("operator")
    if not isinstance(record, Mapping) or not isinstance(payload.get("parameters"), Mapping):
        raise TypeError("an adapter/operator carries `operator`, what it is, and `parameters`, "
                        "its values by layer")
    d = int(lm.model.norm.weight.shape[0])
    if int(record.get("d", d)) != d:
        raise ValueError(f"OPERATOR_WIDTH_MISMATCH: the operator was trained on a residual stream "
                         f"{record['d']} wide, and this model's is {d} wide")
    spec = read_operator_spec({k: record[k] for k in FIELDS if k in record},
                              n_layers=len(lm.model.layers), d=d)
    modules = build_operator_modules(spec, d, values=payload["parameters"])
    for module in modules.values():
        module.freeze()
    return attach_operators(lm, modules)
