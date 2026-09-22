"""Ops over **trajectories**: a residual vector followed along an axis.

The logit lens computes one position's vector at every layer; a story
trace computes one layer's vector at every position. Both are the same
object read along a different axis, and a trajectory record makes that
explicit:

* `axis: "layers"` — one position's vector at every captured layer:
  where in *depth* a commitment forms.
* `axis: "positions"` — one layer's vector at every position along a
  sequence: where in a *text* it forms.

A trajectory is a collection of `trajectory/point` — vector items
(`id`, `coords`, `space`, `vector`, `norm`) with `step`, `position` and
the `token` read — `step` indexing the axis. A projected trajectory is a
collection of `activations/coordinate`: `coord` (the scalar coordinate
along a direction) in place of the vector, which is what makes a
corpus-scale trace small enough to store.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Op

OPS: tuple[Op, ...] = ()
