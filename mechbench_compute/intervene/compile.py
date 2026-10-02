from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.intervene.compiled import Compiled
from mechbench_compute.intervene.constants import SWEEP_AXES
from mechbench_compute.intervene.fill_feature import fill_feature
from mechbench_compute.intervene.operator_refused import OperatorRefused
from mechbench_compute.intervene.read_mask import PORT_WORD
from mechbench_compute.intervene.spec import Spec
from mechbench_compute.intervene.spec_error import SpecError


def compile(model, items: Sequence[Mapping[str, Any]], *,
            inputs: Mapping[str, Any] | None = None, seed: int = 0) -> Compiled:
    inputs = inputs or {}
    port_dir = inputs.get("direction")
    port_src = inputs.get("source")
    filled = []
    port_dictionary = inputs.get("dictionary")
    for it in items:
        it = fill_feature(it, port_dictionary)
        operator = it.get("f") is not None
        if operator and it.get("parameter") is not None:
            raise OperatorRefused("OPERATOR_FIELDS", "an operator acts on an activation, and an item "
                                  "that names a `parameter` edits a weight", construct="parameter")
        if operator and it.get("mask") == PORT_WORD and port_dir is not None:
            it["mask"] = port_dir
        if not operator and it.get("direction") is None and port_dir is not None:
            it["direction"] = port_dir
        constants = it.get("constants")
        binds = operator and isinstance(constants, Mapping) and any(
            isinstance(v, Mapping) for v in constants.values())
        if it.get("source") is None and port_src is not None and (
                it.get("op") in ("mean", "resample", "patch") or binds):
            it["source"] = port_src
        filled.append(it)
    weight_items = [it for it in filled if it.get("parameter") is not None]
    activation_items = [it for it in filled if it.get("parameter") is None]
    for it in activation_items:
        over = it.get("sweep_over")
        if over is None:
            continue
        if not isinstance(over, (list, tuple)) or any(a not in SWEEP_AXES for a in over):
            raise SpecError(
                f"`sweep_over` names sweep axes ({', '.join(SWEEP_AXES)}), "
                f"not {over!r}")
    specs = [Spec(it, n_layers=model.arch.n_layers, seed=int(seed)) for it in activation_items]
    if not specs and not weight_items:
        raise SpecError("an intervention needs a non-empty spec list")
    return Compiled(specs, weight_items, filled, activation_items=activation_items,
                    n_layers=model.arch.n_layers, seed=int(seed))
