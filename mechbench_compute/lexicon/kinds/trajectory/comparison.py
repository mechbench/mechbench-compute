from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F, ID

KIND = Kind(
    "trajectory/comparison",
    "Two trajectories compared at one step: cosine, angle, and the norms.",
    fields={"id": ID, "step": F("integer", "The step."), "layer": F("integer", "The layer."), "position": F("integer", "The position."),
            "cosine": F("number", "Cosine between the two vectors."), "angle_deg": F("number", "The angle in degrees."),
            "norm_a": F("number", "The first vector's norm."), "norm_b": F("number", "The second's."), "norm_ratio": F("number", "b / a.")},
    required=("step", "cosine"),
    key=("id", "step"),
    header={"axis": "The shared axis.", "pair_by": "`id` or `step`.", "threshold": "The cosine below which they count as diverged.",
            "n_pairs": "Rows paired.", "divergence_step": "The first step below the threshold.", "min_cosine_step": "The step of least agreement.",
            "per_step": "`{step, mean_cosine, n}` across ids."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="What `trajectory/compare` produces: the two trajectories paired by id or by step, and at each step the "
        "cosine and angle between their vectors and the ratio of their norms. The header carries the first "
        "step at which the cosine fell below the threshold — where two models, or two prompts, stop agreeing.",
)
