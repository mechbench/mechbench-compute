"""Directions as first-class objects.

A `direction/vector` is a unit vector in a model's activation space
with provenance: how it was made (difference of means, PCA, a probe's
weight, an SAE feature, a trained steering vector, arithmetic over
other directions), from which objects, on which model. It is an
`activations/vector` — `{space, vector, norm}` — plus `unit` and
`derivation`, so one object type flows through `intervene/apply`
(add / project-out / rotate), `direction/project`, attribution and the
vocabulary projection, and a direction found one way can be tried every
other way without conversion.

Producers here are pure (numpy over vector items) except
`unembed_direction`, which needs a model's unembedding.
"""

from __future__ import annotations

from mechbench_compute.directions.add import add  # noqa: F401
from mechbench_compute.directions.coerce_array import coerce_array  # noqa: F401
from mechbench_compute.directions.make import make  # noqa: F401
from mechbench_compute.directions.read_space import read_space  # noqa: F401

#: Pure blocks this package contributes, taken by `blocks.PURE_BLOCKS`.
PURE_DIRECTION_BLOCKS = {
}
