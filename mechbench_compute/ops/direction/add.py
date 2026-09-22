from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import points as P
from mechbench_compute.directions.add import add
from mechbench_compute.directions.directions_from import _directions_from
from mechbench_compute.lexicon._base import Op, Output, P
from mechbench_compute.lexicon.direction import _NAMED_DIRECTIONS

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
    inputs=_NAMED_DIRECTIONS,
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
    return block_add(inputs, params)


def block_add(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return add(_directions_from(inputs, params), params.get("weights"))
