from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute.lexicon._base import In, Op, Otherwise, Output, P
from mechbench_compute.trajectory.header import _header
from mechbench_compute.trajectory.rows import _rows
from mechbench_compute.trajectory.trajectory_of import _trajectory_of

OP = Op(
    name="trajectory/aggregate",
    summary=(
        "Group a trajectory's rows and reduce them — a mean trajectory with "
        "spread per step, one value per group over a window, or per-group "
        "mean vectors that `direction/fit` can read directly."
    ),
    description="""\
Items are grouped `by` a coordinate (`label` by default — which also reads
the older `label` field — or any coordinate such as `genre`), by `id`, or by
a field on the items, optionally restricted to a `steps` window, and
reduced `as`:

* `"per_step"` — for each (group, step): with vectors, the mean vector, its
  norm, the mean norm of the members, and `spread` (mean cosine of members
  to the mean); with coordinates, mean and std.
* `"window"` — one value per group over every step in the window: with
  coordinates, mean/std/n (a story's late-window commitment, with
  `by: "id"`); with vectors, the mean vector.
* `"vectors"` — one `activations/vector` per group, the mean over its
  members and steps in the window, with the group as its coordinate on the
  `by` axis. This is the shape `direction/fit` reads, so an
  outcome axis — "the lighthouse-story mean minus the other-story mean over
  tokens 5 … 30" — is this block followed by that one (with `axis` set to
  the same `by`).
""",
    inputs=(
        In("trajectory", "trajectory/point | activations/coordinate",
           "A trajectory of points, or a projected one of coordinates, read "
           "the same way.", many=True),
    ),
    output=Output('trajectory/summary', collection=True, doc="For `per_step` and `window`: one item per group (and per step), with a mean `vector` or a mean `coord` as the input had; the header repeats the input's and adds `aggregated: {by, as, steps}`. For `vectors`: a collection of `activations/vector` instead, one item per group with the group on the `by` coordinate.",
                 otherwise=(Otherwise("activations/vector", collection=True, param="as", equals="vectors"),)),
    params=(
        P("by", "string",
          "What to group on: a coordinate (`\"label\"`, `\"genre\"`), "
          "`\"id\"`, or a field present on the items.",
          "label"),
        P("as", "string",
          "The reduction: `\"per_step\"`, `\"window\"` or `\"vectors\"` — "
          "see above.",
          "per_step", choices=("per_step", "window", "vectors")),
        P("steps", "\"all\" | object",
          "Which steps take part: `\"all\"` or `{\"range\": [a, b]}`, steps `a` … `b − 1`.",
          "all", fields=(
              P("range", "list[int]", "`[a, b]`: steps `a` … `b − 1`."),
          )),
    ),
    example={
        "by": "label",
        "as": "vectors",
        "steps": {"range": [5, 30]},
    },
    example_inputs={"trajectory": {"$ref": {"bench": "you/lab/trajectory"}}},
)


def run(ctx, inputs, params):
    return aggregate(inputs, params)


def aggregate(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """trajectory/aggregate — group rows and reduce.

    by:      "label" (default) | "id" | a coord/field name on the rows
    steps:   "all" | {"range": [a, b]} — which steps enter the reduce
    as:      "per_step" (default): mean per (group, step) with spread —
                 vectors → mean vector, mean norm, spread (mean cosine
                 to the mean); coords → mean, std.
             "window":  ONE value per group over the step window —
                 coords → mean/std/n (014's late-window commitment per
                 story, with by: "id"); vectors → mean vector.
             "vectors": one `residual_vectors` row per group (mean over
                 items and steps in the window), labelled by the group —
                 what `direction/fit` reads, so an outcome axis
                 is this block followed by that one.
    """
    traj = _trajectory_of(inputs.get("trajectory"),
                          coords_ok=True)
    by = str(params.get("by", "label"))
    mode = str(params.get("as", "per_step"))
    if mode not in ("per_step", "window", "vectors"):
        raise ValueError("`as` must be 'per_step', 'window' or 'vectors'")
    steps = params.get("steps", "all")
    lo, hi = (None, None)
    if isinstance(steps, Mapping) and "range" in steps:
        lo, hi = int(steps["range"][0]), int(steps["range"][1])
    rows = _rows(traj)
    scalar = bool(rows) and "coord" in rows[0]
    if mode == "vectors" and scalar:
        raise ValueError("as: 'vectors' needs vector rows, not a projection")

    def group_of(r):
        # A coordinate first (`by: "genre"`), `id`, then a field on the
        # item; the retired `label` field is the `label` coordinate.
        if by == "id":
            return r.get("id")
        g = S.label_of(r, by)
        return g if g is not None else r.get(by)

    groups: dict[Any, dict[int, list]] = {}
    for r in rows:
        s = int(r["step"])
        if lo is not None and not (lo <= s < hi):
            continue
        g = group_of(r)
        val = r["coord"] if scalar else np.asarray(r["vector"], dtype=np.float32)
        groups.setdefault(g, {}).setdefault(s, []).append(val)
    if not groups:
        raise ValueError("no rows to aggregate (empty window or no groups)")

    out_rows: list[dict[str, Any]] = []
    if mode == "per_step":
        for g, by_s in groups.items():
            for s in sorted(by_s):
                vals = by_s[s]
                if scalar:
                    a = np.asarray(vals, dtype=np.float64)
                    out_rows.append({"group": g, "step": s, "n": len(vals),
                                     "mean": round(float(a.mean()), 6),
                                     "std": round(float(a.std()), 6)})
                else:
                    m = np.mean(np.stack(vals), axis=0)
                    mn = float(np.linalg.norm(m))
                    cos = [float(v @ m / (np.linalg.norm(v) * mn))
                           for v in vals if mn and np.linalg.norm(v)]
                    out_rows.append({
                        "group": g, "step": s, "n": len(vals),
                        "norm": round(mn, 5),
                        "mean_norm": round(float(np.mean(
                            [np.linalg.norm(v) for v in vals])), 5),
                        "spread": round(float(np.mean(cos)), 6) if cos else None,
                        "vector": [round(float(x), 5) for x in m],
                    })
        from mechbench_compute.lexicon import kinds as K

        return K.collection("trajectory/summary", out_rows, **_header(traj),
                            aggregated={"by": by, "as": mode, "steps": steps})

    # window / vectors: one value per group over everything in the window
    first_space = S.space_of(rows[0], header=traj) if rows else None
    layer = first_space.get("layer") if first_space else None
    for g, by_s in groups.items():
        vals = [v for s in by_s for v in by_s[s]]
        if scalar:
            a = np.asarray(vals, dtype=np.float64)
            out_rows.append({"group": g, "n": len(vals),
                             "mean": round(float(a.mean()), 6),
                             "std": round(float(a.std()), 6)})
        else:
            m = np.mean(np.stack(vals), axis=0)
            # The group goes out as a string coordinate on the `by` axis:
            # `direction/fit` names its groups as strings, and a
            # text/measure hit arrives as the integer 1/0.
            out_rows.append(S.vector(m, first_space, id=str(g), coords={by: str(g)},
                                     n_pooled=len(vals)))
    from mechbench_compute.lexicon import kinds as K

    if mode == "vectors":
        return K.collection(
            "activations/vector", out_rows,
            model=first_space.get("model") if first_space else None,
            point=traj.get("point"),
            source="resid",
            position=f"trajectory-window {steps}",
            layers=[layer] if layer is not None else traj.get("layers"),
            d_model=traj.get("d_model"),
            template=traj.get("template"),
        )
    return K.collection("trajectory/summary", out_rows, **_header(traj),
                        aggregated={"by": by, "as": mode, "steps": steps})
