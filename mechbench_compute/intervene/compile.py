from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.intervene.compiled import Compiled
from mechbench_compute.intervene.constants import SWEEP_AXES
from mechbench_compute.intervene.spec import Spec
from mechbench_compute.intervene.spec_error import SpecError


def compile(model, items: Sequence[Mapping[str, Any]], *,
            inputs: Mapping[str, Any] | None = None, seed: int = 0) -> Compiled:
    """Parse spec items for `model`. Objects may arrive by edge: a
    `direction` / `source` port in `inputs` fills any item that names none
    of its own. An item that names a `parameter` edits a WEIGHT, not an
    activation (task 000457): its scope is the node rather than the
    forward pass — the tensor is changed, every record runs against the
    changed model, and the original is reinstalled afterwards. The two
    kinds compose, so they are separated here and applied in their own
    scopes. Shared by `intervene/apply` and by the text ops that take an
    intervention (000601)."""
    inputs = inputs or {}
    port_dir = inputs.get("direction")
    port_src = inputs.get("source")
    filled = []
    for it in items:
        it = dict(it)
        if it.get("direction") is None and port_dir is not None:
            it["direction"] = port_dir
        if it.get("source") is None and port_src is not None and it.get("op") in ("mean", "resample", "patch"):
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
