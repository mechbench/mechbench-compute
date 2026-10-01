from __future__ import annotations

from mechbench_compute.lexicon._base import Kind, Metric
from mechbench_compute.lexicon.values import COORDS, F, ID

KIND = Kind(
    "adapter/delta",
    "What training wrote at one module: the norm, spectrum and effective rank of an adapter's delta there, and its share of the adapter's mass.",
    extends="records/record",
    fields={"id": ID, "coords": COORDS,
            "frobenius": F("number", "‖ΔW‖_F at this module — how much was written here at all."),
            "spectral": F("number", "The largest singular value: how much of the write is one direction."),
            "singular_values": F("array", "The spectrum, largest first (as many as `adapter/measure`'s `k`).", items={"type": "number"}),
            "effective_rank": F("number", "exp(H(p)) over the normalised spectrum: 1 for a single direction, r for r equal ones."),
            "mass_share": F("number", "This module's ‖ΔW‖² over the total of what was measured; the shares sum to 1."),
            "rank": F("integer", "The adapter's rank at this module — the number of directions it could have written."),
            "shape": F("array", "`[out, in]` of the module this delta is over.", items={"type": "integer"}),
            "vector": F("array", "The principal left-singular direction, unit length, in the module's output space — present when the node asked for it.", items={"type": "number"}),
            "basis": F("object", "`{module, side, d}` — which space `vector` lives in.")},
    required=("id", "frobenius", "effective_rank", "mass_share"),
    key=("id", "coords"),
    header={"base_model": "The base the adapter was trained on.",
            "trained_on": "`{base, adapters}` — the full stack.",
            "lora": "`{rank, alpha, scale, target_modules}` of the adapter read.",
            "measured": "`{modules, layers, frobenius}` — what this node covered, which is what the shares are shares of.",
            "source": "What to call this adapter, stamped on every item's `coords.adapter`."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="What `adapter/measure` produces: one item per (layer, module), read from the adapter's own low-rank factors "
        "with no model and no forward pass. `coords.layer` and `coords.module` are what a reading groups on — "
        "mass by layer is the 'where did training write' map — and `coords.adapter` separates two adapters "
        "measured into one collection. With `vectors` on, two adapters' writes at the same module are compared "
        "by `geometry/compare` through their principal directions — whether two training runs moved the model "
        "the same way, answered in weight space rather than by capturing what they do to a prompt.",
    metrics=(
        Metric("cosine", "similarity", True,
               "The cosine between the two deltas' principal directions, in [−1, 1]. Needs `vectors` on the "
               "measuring node, and compares only within one module — group by `coords.module` (`by: \"module\"`), "
               "since two modules' output spaces are different spaces."),
    ),
)
