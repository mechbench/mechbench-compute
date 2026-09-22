"""Residual trajectories as objects.

A position's vector at every layer and a layer's vector at every
position are the same object read along a different axis, and that
object is a kind here:

    axis "layers"     one POSITION's vector at every captured layer —
                      the funnel: where in depth a commitment forms.
    axis "positions"  one LAYER's vector at every position along a
                      sequence — the trace: where in a story it forms.

A trajectory is a collection of `trajectory/point` — vector items
(`space`, `vector`, `norm`) with `step`, `position` and the token read —
`step` indexing the axis. A projected one is a collection of
`activations/coordinate`. Four blocks:

    trajectory/capture    the model block: capture one per record.
    trajectory/project    scalar coordinate of every row along a
                          direction, for a trace to read.
    trajectory/compare    two trajectories → per-step cosine, angle,
                          norm ratio, and the divergence step.
    trajectory/aggregate  group rows and reduce: mean trajectory + spread
                          per step; or a windowed mean per group emitted
                          as `residual_vectors`, so `direction/fit`
                          reads it unchanged — which is how an axis like
                          "one group's mean minus another's over tokens
                          5..30" is built.

REPLAY. A stored corpus at trace fidelity carries the exact token ids a
story was generated as (`trace.token_ids`) and where generation began
(`trace.generation_spans`). `replay: "auto"` (the default) reads those
when present and tokenizes the text otherwise — re-tokenizing a
generated story can shift a boundary, and the trace is the ground
truth of what the model actually saw.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mechbench_compute.trajectory.project import project  # noqa: F401

#: Pure blocks this package contributes, taken by `blocks.PURE_BLOCKS`.
PURE: dict[str, Callable[..., Any]] = {
}
