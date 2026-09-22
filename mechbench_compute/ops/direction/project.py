from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.directions.as_array import as_array
from mechbench_compute.directions.items_at import _items_at
from mechbench_compute.lexicon._base import In, Op, Output

OP = Op(
    name="direction/project",
    summary=(
        "Project every vector of a collection onto a direction — each "
        "prompt's scalar coordinate along that axis."
    ),
    description="""\
For each item of the collection at the direction's layer, the dot product
of the item's vector with the unit direction. Each coordinate keeps the
item's `id`, `coords` and `space`, so the result groups and plots the same
way the vectors did. A quick way to see whether a direction separates the
groups it was built from — or ones it was not.
""",
    inputs=(
        In("vectors", "activations/vector",
           "A collection of vectors with items at the direction's layer.",
           many=True),
        In("direction", "direction/vector", "The direction to project onto."),
    ),
    output=(
        Output('activations/coordinate', collection=True, doc="One item per input vector: `id`, `coords`, `space`, the `direction`'s identity and `coord`, the dot product with the unit direction.")
    ),
    params=(),
    example={},
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/vectors"}}, "direction": {"$ref": {"bench": "you/lab/axis"}}},
)


def run(ctx, inputs, params):
    return block_project(inputs, params)


def project_rows(vectors: Mapping[str, Any], d: Mapping[str, Any]) -> dict[str, Any]:
    """Each item of a vector collection at the direction's layer,
    projected onto the direction: the scalar coordinate along it."""
    layer = S.layer_of(d)
    rows = _items_at(vectors, layer)
    u = as_array(d)
    out = []
    for r in rows:
        v = np.asarray(r["vector"], dtype=np.float32)
        if v.size != u.size:
            raise ValueError("vector width does not match the direction")
        out.append(S.coordinate(float(v @ u), S.space_of(r, header=vectors), d,
                                id=r.get("id"), coords=S.coords_of(r),
                                token=r.get("token")))
    from mechbench_compute.lexicon import kinds as K

    return K.collection("activations/coordinate", out, projected=True)


def block_project(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return project_rows(inputs.get("vectors"),
                        inputs.get("direction"))
