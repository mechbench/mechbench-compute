from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.read_expr_params import read_expr_params


def evaluate_param_expr(src: str, bound_params: Mapping[str, Any]) -> Any:
    from mechbench_compute.expr.engine import load_engine

    scope = {k: bound_params[k] for k in read_expr_params(src) if k in bound_params}
    got = load_engine().evaluate(src, None, scope)
    if got.undefined:
        reasons = ", ".join(sorted(got.undefined))
        raise ValueError(f"{{\"$expr\": {src!r}}} has no value: {reasons}")
    return got.values[0]
