from __future__ import annotations

from mechbench_compute import points as P
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.trajectory.project import project

_TRAJECTORY = In("trajectory", "trajectory/point",
                 "A trajectory: a collection of points with vectors.", many=True)


OP = Op(
    name="trajectory/project",
    summary=(
        "Project every step of a trajectory onto a direction — the trace of "
        "a text, or the funnel of a prompt, as one number per step."
    ),
    description="""\
Each point's vector becomes its dot product with the unit direction,
recorded as `coord` on an `activations/coordinate` item that keeps the
point's `step`, `position`, `token` and `space`; the vectors are dropped
unless `keep_vectors` is set. The direction's layer is recorded but not
enforced: projecting a layers-axis trajectory onto a single-layer direction
is the funnel read against one axis, which is a legitimate question.
""",
    inputs=(
        _TRAJECTORY,
        In("direction", "direction/vector",
           "A direction of the trajectory's width."),
    ),
    output=(
        Output('activations/coordinate', collection=True, doc="One coordinate per input point: `id`, `coords`, `space`, `step`, `position`, `token`, the `direction`'s identity and `coord` (the vector kept as well under `keep_vectors`); the header repeats the input's, with `projected: true`.")
    ),
    params=(
        P("keep_vectors", "bool",
          "Keep each row's `vector` beside its `coord`.",
          False),
    ),
    example={"keep_vectors": False},
    example_inputs={
        "trajectory": {"$ref": {"bench": "you/lab/trajectory"}},
        "direction": {"$ref": {"bench": "you/lab/axis"}},
    },
)


def run(ctx, inputs, params):
    return project(inputs, params)
