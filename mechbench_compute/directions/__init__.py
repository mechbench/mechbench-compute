"""Directions as first-class objects (task 000367, epic 000364).

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
`vocab_projection`, which needs a model's unembedding.
"""

from __future__ import annotations

import hashlib

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute.directions.as_array import as_array  # noqa: F401
from mechbench_compute.directions.constants import KIND  # noqa: F401
from mechbench_compute.directions.is_direction import _is_direction  # noqa: F401
from mechbench_compute.directions.add import add  # noqa: F401
from mechbench_compute.directions.constants import DEFAULT_AXIS  # noqa: F401
from mechbench_compute.directions.directions_from import _directions_from  # noqa: F401
from mechbench_compute.directions.items_at import _items_at  # noqa: F401
from mechbench_compute.directions.make import make  # noqa: F401
from mechbench_compute.directions.model_provenance import _model_provenance  # noqa: F401
from mechbench_compute.directions.models_of import _models_of  # noqa: F401
from mechbench_compute.directions.same_space import same_space  # noqa: F401
from mechbench_compute.directions.space_at import _space_at  # noqa: F401
from mechbench_compute.directions.space_of import space_of  # noqa: F401


# --- construction ----------------------------------------------------------


# --- producers --------------------------------------------------------------


# --- arithmetic (pure) --------------------------------------------------------


# --- the unembedding as a lens -------------------------------------------------------


# --- pure-block adapters (inputs, params) ---------------------------------------------


PURE_DIRECTION_BLOCKS = {
}
