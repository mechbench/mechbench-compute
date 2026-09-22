from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.trajectory.rows import _rows
from mechbench_compute.trajectory.trajectory_of import _trajectory_of

OP = Op(
    name="trajectory/compare",
    summary=(
        "Compare two trajectories step by step — cosine, angle and norm "
        "ratio at each step, and the step at which they diverge."
    ),
    description="""\
Rows are paired by (`id`, `step`) — the same texts under two models, or two
conditions — or by `step` alone (`pair_by: "step"`) for two single-item
trajectories. For each pair the block reports the cosine between the two
vectors, the angle in degrees, both norms and their ratio. Across pairs it
reports the mean cosine per step and the **divergence step**: the first
step at which the mean cosine falls below `threshold`.

Both trajectories must have the same axis and carry vectors (not
projections).
""",
    inputs=(
        In("a", "trajectory/point", "The first trajectory.", many=True),
        In("b", "trajectory/point", "The second trajectory.", many=True),
    ),
    output=Output('trajectory/comparison', collection=True, doc='One item per pair: `cosine`, `angle_deg`, `norm_a`, `norm_b`, `norm_ratio`. The header carries `divergence_step`, `min_cosine_step`, `per_step` (`{step, mean_cosine, n}`) and `n_pairs`.'),
    params=(
        P("pair_by", "string",
          "`\"id\"`: pair rows with the same record id and step. `\"step\"`: "
          "pair by step alone.",
          "id", choices=("id", "step")),
        P("threshold", "float",
          "The mean cosine below which the trajectories count as diverged.",
          0.9),
    ),
    example={"threshold": 0.9},
    example_inputs={"a": {"$ref": {"bench": "you/lab/base_traj"}}, "b": {"$ref": {"bench": "you/lab/tuned_traj"}}},
)


def run(ctx, inputs, params):
    return compare(inputs, params)


def compare(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """trajectory/compare — trajectories `a` and `b`,
    paired by (id, step) (or by step alone with `pair_by: "step"`, for
    two single-item trajectories under different prompts or models):
    per-step cosine, angle in degrees, norm ratio; and the DIVERGENCE
    step — the first at which cosine falls below `threshold`."""
    a = _trajectory_of(inputs.get("a"), "a")
    b = _trajectory_of(inputs.get("b"), "b")
    if a.get("axis") != b.get("axis"):
        raise ValueError("trajectories must share an axis to be compared")
    pair_by = str(params.get("pair_by", "id"))
    if pair_by not in ("id", "step"):
        raise ValueError(f"pair_by must be 'id' or 'step', not {pair_by!r}")
    threshold = float(params.get("threshold", 0.9))

    def key(r):
        return (r.get("id"), r["step"]) if pair_by == "id" else (r["step"],)

    bm = {key(r): r for r in _rows(b)}
    rows = []
    for r in _rows(a):
        s = bm.get(key(r))
        if s is None:
            continue
        va = np.asarray(r["vector"], dtype=np.float32)
        vb = np.asarray(s["vector"], dtype=np.float32)
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        cos = float(va @ vb / (na * nb)) if na and nb else 0.0
        cos = max(-1.0, min(1.0, cos))
        rows.append({
            "id": r.get("id") if pair_by == "id" else None,
            "step": r["step"], "layer": r.get("layer"),
            "position": r.get("position"),
            "cosine": round(cos, 6),
            "angle_deg": round(math.degrees(math.acos(cos)), 4),
            "norm_a": round(na, 5), "norm_b": round(nb, 5),
            "norm_ratio": round(nb / na, 6) if na else None,
        })
    if not rows:
        raise ValueError("no rows paired — do the trajectories share ids/steps?")
    # Per-step summary across ids, then the divergence step over it.
    by_step: dict[int, list[float]] = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r["cosine"])
    steps = sorted(by_step)
    mean_cos = {s: float(np.mean(by_step[s])) for s in steps}
    diverge = next((s for s in steps if mean_cos[s] < threshold), None)
    from mechbench_compute.lexicon import kinds as K

    return K.collection(
        "trajectory/comparison", rows,
        axis=a.get("axis"),
        pair_by=pair_by,
        threshold=threshold,
        n_pairs=len(rows),
        divergence_step=diverge,
        min_cosine_step=min(steps, key=lambda s: mean_cos[s]),
        per_step=[{"step": s, "mean_cosine": round(mean_cos[s], 6),
                   "n": len(by_step[s])} for s in steps],
    )
