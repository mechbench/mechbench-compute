"""The declarative `intervene` block (task 000366, epic 000364).

Intervention is a product of two small sets. POINTS: any hook point of
the forward pass. OPERATIONS: zero, mean, resample, patch, add, scale,
clamp, project_out, rotate — over sets of positions, heads and
neurons, optionally conditioned on a direction's projection or on
token identity. One spec list, applied together in one forward, then
a readout (the next-token distribution, or captures). The old
ablate/steer blocks are special cases of this grammar.

A spec item::

    {"point": "resid_post", "layers": [14] | "all",
     "positions": "last" | "all" | [3, 5] | {"tokens": ["lighthouse"]} | {"range": [2, 6]},
     "heads": [0, 3] | null, "neurons": [17, 902] | null,
     "op": "add", "strength": 4.0,
     "direction": <direction object>, "direction2": <direction> (rotate),
     "source": <residual_vectors record> (mean / resample / patch), "row": {...},
     "condition": {"direction": <direction>, "threshold": 0.0, "above": true}}

Everything here is deterministic given the spec and the records
(`resample` draws from `seed`), so the block is `reproducible`.
"""

from __future__ import annotations

import contextlib
import json

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from mechbench_compute import directions as dirs
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute.points import LAYOUT as _LAYOUT
from mechbench_compute.intervene.coerce_int_list import coerce_int_list  # noqa: F401
from mechbench_compute.intervene.coerce_axis_coord import coerce_axis_coord  # noqa: F401
from mechbench_compute.intervene.cell import Cell  # noqa: F401
from mechbench_compute.intervene.compile import compile  # noqa: F401
from mechbench_compute.intervene.compiled import Compiled  # noqa: F401
from mechbench_compute.intervene.constants import SWEEP_AXES  # noqa: F401
from mechbench_compute.intervene.edit_weights import edit_weights  # noqa: F401
from mechbench_compute.intervene.plan import Plan, plan  # noqa: F401
from mechbench_compute.intervene.build_rows_matrix import build_rows_matrix  # noqa: F401
from mechbench_compute.intervene.scale_specs import scale_specs  # noqa: F401
from mechbench_compute.intervene.read_source_items import read_source_items  # noqa: F401
from mechbench_compute.intervene.spec import OPS, Spec, _GLOBAL_POINTS, _SAME  # noqa: F401
from mechbench_compute.intervene.spec_error import SpecError  # noqa: F401
from mechbench_compute.intervene.spec_intervention import SpecIntervention  # noqa: F401
from mechbench_compute.intervene.read_spec_items import read_spec_items  # noqa: F401
from mechbench_compute.intervene.sweep_as_run import sweep_as_run  # noqa: F401
from mechbench_compute.intervene.sweep_cells import sweep_cells  # noqa: F401
from mechbench_compute.intervene.serialize_spec import serialize_spec  # noqa: F401


# --- parsing -----------------------------------------------------------------------


def sweep_factors(params: Mapping[str, Any]) -> list[float]:
    """The strengths a node's `sweep` runs — `sweep_cells` read by a
    caller that varies nothing else."""
    return [c.factor for c in sweep_cells(params)]


# --- the block ---------------------------------------------------------------------


