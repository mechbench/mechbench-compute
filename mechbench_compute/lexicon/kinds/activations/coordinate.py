from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import COORDS, F, ID, SPACE, TOKEN

KIND = Kind(
    "activations/coordinate",
    "A vector's scalar coordinate along a direction, with the space and the direction's identity.",
    fields={"id": ID, "coords": COORDS, "space": SPACE,
            "direction": F("object", "`{space, method, …}` — the direction projected onto, without its vector."),
            "coord": F("number", "The dot product with the unit direction."),
            "step": F("integer", "The step, when read along a trajectory."),
            "position": F("integer", "The position, when read along a trajectory."),
            "token": TOKEN},
    required=("space", "direction", "coord"),
    key=("id", "step", "space"),
    header={"axis": "For a projected trajectory: `layers` or `positions`.",
            "projected": "Always true: a coordinate collection is a projected one."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="What `direction/project` and `trajectory/project` emit: one number per vector, the dot product with a "
        "unit direction, with the space and the direction's derivation carried so the number can be read back "
        "to what it measures. A projected trajectory is a collection of these keyed by step; the vector itself "
        "is not kept, which is the point.",
)
