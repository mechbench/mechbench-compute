"""Residual trajectories as objects (task 000368, lexicon epic 000364).

The lens computes a position's vector at every layer implicitly and
throws it away; experiment 014 computed a layer's vector at every
position along a story by hand. Both are the same object read along a
different axis, and this module makes it a kind:

    axis "layers"     one POSITION's vector at every captured layer —
                      the funnel (013), where in depth a commitment forms.
    axis "positions"  one LAYER's vector at every position along a
                      sequence — the trace (014), where in a story it
                      forms.

A trajectory is a collection of `trajectory/point` — vector items
(`space`, `vector`, `norm`) with `step`, `position` and the token read —
`step` indexing the axis. A projected one is a collection of
`activations/coordinate`. Four blocks:

    trajectory/capture    the model block: capture one per record.
    trajectory/project    scalar coordinate of every row along a
                          direction (directions.py), for a trace to read.
    trajectory/compare    two trajectories → per-step cosine, angle,
                          norm ratio, and the divergence step.
    trajectory/aggregate  group rows and reduce: mean trajectory + spread
                          per step; or a windowed mean per group emitted
                          as `residual_vectors`, so `direction/fit`
                          reads it unchanged (014's outcome axis is
                          "lighthouse-story mean minus other-story mean
                          over tokens 5..30" — exactly that).

REPLAY. A stored corpus at trace fidelity carries the exact token ids a
story was generated as (`trace.token_ids`) and where generation began
(`trace.generation_spans`). `replay: "auto"` (the default) reads those
when present and tokenizes the text otherwise — re-tokenizing a
generated story can shift a boundary, and the trace is the ground
truth of what the model actually saw.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute.trajectory.header import _header  # noqa: F401
from mechbench_compute.trajectory.project import project  # noqa: F401
from mechbench_compute.trajectory.rows import _rows  # noqa: F401
from mechbench_compute.trajectory.trajectory_of import _trajectory_of  # noqa: F401

# --- shapes ------------------------------------------------------------------


# --- capture (the model block) -----------------------------------------------


# --- pure blocks over trajectories -------------------------------------------


#: Pure blocks this module contributes (registered by blocks.PURE_BLOCKS).
PURE: dict[str, Callable[..., Any]] = {
}
