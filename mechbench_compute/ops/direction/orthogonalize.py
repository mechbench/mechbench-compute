from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute.directions.coerce_array import coerce_array
from mechbench_compute.directions.make import make
from mechbench_compute.directions.check_same_space import check_same_space
from mechbench_compute.directions.read_space import read_space
from mechbench_compute.lexicon._base import In, Op, Output

OP = Op(
    name="direction/orthogonalize",
    summary=(
        "Remove from a direction its components along one or more other "
        "directions — what is left of a concept once a confound is taken "
        "out."
    ),
    description="""\
The `against` directions are orthonormalised (Gram–Schmidt) and the
direction's projection onto each is subtracted; the remainder is
re-normalised. A direction that lies entirely within the span of `against`
has nothing left and the block refuses it. All inputs must share a space.
""",
    inputs=(
        In("direction", "direction/vector", "The direction to clean."),
        In("against", "direction/vector",
           "The direction(s) to remove — one, or a list of them.", many=True),
    ),
    output=(
        Output('direction/vector', collection=False, doc='`derivation.method` is `"orthogonalize"`, with `derivation.against` (how many independent directions were removed).')
    ),
    params=(),
    example={},
    example_inputs={
        "direction": {"$ref": {"bench": "you/lab/sentiment"}},
        "against": [{"$ref": {"bench": "you/lab/length"}}],
    },
)


def run(ctx, inputs, params):
    d = inputs.get("direction")
    against = inputs.get("against")
    if isinstance(against, Mapping):
        against = [against]
    return orthogonalize(d, list(against or []))


def orthogonalize(d: Mapping[str, Any],
                  against: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    v = coerce_array(d)
    basis: list[np.ndarray] = []
    for a in against:
        check_same_space(d, a)
        u = coerce_array(a)
        for b in basis:
            u = u - float(u @ b) * b
        n = float(np.linalg.norm(u))
        if n > 1e-8:
            basis.append(u / n)
    for b in basis:
        v = v - float(v @ b) * b
    if float(np.linalg.norm(v)) < 1e-8:
        raise ValueError("direction lies entirely in the span of `against`")
    return make(v, read_space(d), method="orthogonalize", extra={"against": len(basis)})
