from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import DIST, F

KIND = Kind(
    "trajectory/point",
    "One step of a trajectory: the residual vector at one layer and position of one record.",
    extends="activations/vector",
    fields={"step": F("integer", "The step along the axis."),
            "position": F("integer", "The position read."),
            "vocab": DIST,
            "steps": F("array", "The window pooled, when reduced.", items={"type": "integer"})},
    required=("id", "step", "space", "vector"),
    key=("id", "step"),
    header={"axis": "`layers` or `positions`.", "point": "The residual read.", "layers": "The layers.",
            "position": "For a layers axis: which position.", "positions": "For a positions axis: which positions.",
            "d_model": "The vector width.",
            "replay": "`trace`, `text` or `mixed`.", "n_items": "How many records.",
            "max_steps": "The per-record step cap, when set.", "reduce": "The reduction, when reduced.", "steps": "The window, when reduced."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="A projected trajectory is a collection of `activations/coordinate` with `step` and `position`, not a point without its vector.",
)
