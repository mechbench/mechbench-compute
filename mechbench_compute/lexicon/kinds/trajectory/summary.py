from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F, VEC

KIND = Kind(
    "trajectory/summary",
    "A group of trajectory rows reduced at one step, or over a window: the mean vector with its spread, or the mean and std of coordinates.",
    fields={"group": F("string", "The group."), "step": F("integer", "The step, for a per-step summary."), "n": F("integer", "Rows in the group."),
            "mean": F("number", "Mean coordinate."), "std": F("number", "Its standard deviation."),
            "norm": F("number", "The mean vector's norm."), "mean_norm": F("number", "Mean norm of the members."),
            "spread": F("number", "Mean cosine of members to the mean."), "vector": VEC},
    required=("group", "n"),
    key=("group", "step"),
    header={"aggregated": "`{by, as, steps}` — how the rows were grouped and reduced."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="What `trajectory/aggregate` produces, in one of three shapes the header's `as` names: per step, the mean "
        "coordinate and its standard deviation across the group; over a window, one value per group; or as "
        "vectors, the group's mean vector with the spread of its members around it — the shape "
        "`direction/fit` reads directly.",
)
