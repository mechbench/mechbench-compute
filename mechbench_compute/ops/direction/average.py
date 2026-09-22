from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.directions.add import add
from mechbench_compute.directions.directions_from import _directions_from
from mechbench_compute.lexicon._base import Op, Output
from mechbench_compute.lexicon.direction import _NAMED_DIRECTIONS

OP = Op(
    name="direction/average",
    summary=(
        "The mean of several unit directions in the same space — the "
        "component they share."
    ),
    description="""\
Each input is already unit length, so the mean weights every direction
equally however large its original norm was. That is the right question for
"what do these adapters' axes have in common?", where the answer must not be
dominated by whichever axis happened to be longest. Same as `direction/add`
with equal weights, except that the derivation says `average`.
""",
    inputs=_NAMED_DIRECTIONS,
    output=Output('direction/vector', collection=False, doc='`derivation.method` is `"average"`.'),
    params=(),
    example={},
    example_inputs={"directions": [{"$ref": {"bench": "you/lab/axis_a"}}, {"$ref": {"bench": "you/lab/axis_b"}}]},
)


def run(ctx, inputs, params):
    return block_average(inputs, params)


def average(directions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The equal-weight mean of several directions: the shared component.

    Distinct from `add` only in intent and in what the derivation
    records, since both normalize — but the question "what do these
    adapters have in common?" is answered by the mean of their UNIT
    directions, which weights each one equally however long it is.
    """
    out = add(directions)
    out["derivation"]["method"] = "average"
    return out


def block_average(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return average(_directions_from(inputs, params))
