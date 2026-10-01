from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute.directions.add import add
from mechbench_compute.directions.coerce_array import coerce_array
from mechbench_compute.directions.collect_directions import collect_directions
from mechbench_compute.directions.drop_dimensions import drop_dimensions
from mechbench_compute.directions.read_exclude import read_exclude
from mechbench_compute.directions.take_one import take_one
from mechbench_compute.lexicon._base import WILDCARD, In, Op, Output, P

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

`exclude` leaves named dimensions out of every input before the mean:
each is zeroed there and renormalised, so the inputs still count
equally and a cosine with the result agrees with one taken by hand over
the other dimensions. Leave out a model's massive-activation dimension
(Gemma 3's 443) when the inputs were fit without doing so; otherwise
the mean is pulled toward whichever input leans on it most. Centring
does not apply here: it belongs to the captures a direction is fit
from (`direction/fit`'s `center`), not to the direction.
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
    output=Output('direction/vector', collection=False, doc='`derivation.method` is `"average"`, with `exclude` when dimensions were left out.'),
    params=(
        P("exclude", "list[int]",
          "Dimensions left out of every input: zeroed, the input renormalised, and recorded on the "
          "derivation.",
          None),
    ),
    example={},
    example_inputs={"directions": [{"$ref": {"bench": "you/lab/axis_a"}}, {"$ref": {"bench": "you/lab/axis_b"}}]},
)


def run(ctx, inputs, params):
    return average(collect_directions(inputs, params), exclude=params.get("exclude"))


def average(directions: Sequence[Mapping[str, Any]],
            exclude: Sequence[int] | None = None) -> dict[str, Any]:
    dropped = read_exclude(exclude, coerce_array(directions[0]).size) if directions else []
    if dropped:
        directions = [_drop_from(d, dropped) for d in directions]
    out = add(directions)
    out["derivation"]["method"] = "average"
    if dropped:
        out["derivation"]["exclude"] = dropped
    return out


def _drop_from(direction: Mapping[str, Any], dropped: list[int]) -> dict[str, Any]:
    d = dict(take_one(direction))
    v = drop_dimensions(coerce_array(d), dropped)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError(f"a direction lies wholly in the excluded dimensions {dropped}")
    d["vector"] = [float(x) for x in v / n]
    return d
