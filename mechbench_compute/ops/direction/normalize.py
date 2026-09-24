from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.directions.coerce_array import coerce_array
from mechbench_compute.directions.make import make
from mechbench_compute.directions.read_space import read_space
from mechbench_compute.lexicon._base import In, Op, Output

OP = Op(
    name="direction/normalize",
    summary=(
        "Rescale a direction to unit length, keeping its space and model — "
        "an explicit, recorded step for a vector that arrived some other way."
    ),
    description="""\
Directions made by the other `direction/*` ops are unit already. This is
for one that was hand-built or imported, and for making normalisation a
visible step in the graph rather than an assumption.
""",
    inputs=(In("direction", "direction/vector", "The direction to normalise."),),
    output=Output('direction/vector', collection=False, doc='`derivation.method` is `"normalize"`.'),
    params=(),
    example={},
    example_inputs={"direction": {"$ref": {"bench": "you/lab/imported"}}},
)


def run(ctx, inputs, params):
    return normalize(inputs.get("direction"))


def normalize(d: Mapping[str, Any]) -> dict[str, Any]:
    return make(coerce_array(d), read_space(d), method="normalize")
