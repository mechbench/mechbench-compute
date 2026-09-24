from __future__ import annotations

from typing import Any


def arch_header(arch: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"n_layers": int(arch.n_layers)}
    globals_ = getattr(arch, "global_layers", None)
    if globals_ is not None:
        out["global_layers"] = [int(i) for i in globals_]
    first_shared = getattr(arch, "first_kv_shared_layer", None)
    if first_shared is not None:
        out["first_kv_shared_layer"] = int(first_shared)
    return out


from mechbench_compute.blocks.expand_cells import expand_cells  # noqa: F401
from mechbench_compute.blocks.expand_grid import expand_grid  # noqa: F401
from mechbench_compute.blocks.read_group_key import read_group_key  # noqa: F401
from mechbench_compute.blocks.read_items import read_items  # noqa: F401
