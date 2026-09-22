from __future__ import annotations

from mechbench_compute.directions.add import add
from mechbench_compute.directions.collect_directions import collect_directions
from mechbench_compute.lexicon._base import WILDCARD, In, Op, Output, P

OP = Op(
    name="direction/add",
    summary=(
        "Combine several directions in the same space into one by weighted "
        "sum, re-normalised — steer along two concepts at once."
    ),
    description="""\
All inputs must share a layer, point and width; adding across spaces is
meaningless and is refused rather than producing a plausible-looking vector.
`weights` sets the mix (one per direction, default equal), so
`weights: [1, -0.5]` is "the first, minus half the second". The result is
normalised to unit length.
""",
    inputs=(
        In("directions", "direction/vector",
           "The directions as one list, when they do not each arrive on a port "
           "of their own; they are named `d0`, `d1`, … in the result.",
           many=True, required=False),
        In(WILDCARD, "direction/vector",
           "One direction per edge, on a port of your naming; the port name is "
           "the direction's name in the result.", required=False),
    ),
    output=(
        Output('direction/vector', collection=False, doc='`derivation.method` is `"add"`, with `derivation.weights`.')
    ),
    params=(
        P("weights", "list[float]",
          "One coefficient per direction, in input order.",
          None),
    ),
    example={"weights": [1.0, 0.5]},
    example_inputs={"directions": [{"$ref": {"bench": "you/lab/formal"}}, {"$ref": {"bench": "you/lab/terse"}}]},
)


def run(ctx, inputs, params):
    return add(collect_directions(inputs, params), params.get("weights"))

