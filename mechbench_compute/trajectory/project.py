from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.trajectory.header import _header
from mechbench_compute.trajectory.rows import _rows
from mechbench_compute.trajectory.trajectory_of import _trajectory_of


def project(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """trajectory/project — every row's scalar coordinate
    along a direction. The direction's layer is recorded, not enforced:
    projecting a layers-axis trajectory onto a single-layer direction is
    the funnel read against one axis, which is a legitimate question."""
    from mechbench_compute import directions as dirs

    traj = _trajectory_of(inputs.get("trajectory"))
    d = inputs.get("direction")
    if not isinstance(d, Mapping) or "vector" not in d:
        raise ValueError("trajectory/project needs a `direction` record")
    dv = dirs.as_array(d)
    if dv.shape[0] != int(traj.get("d_model", dv.shape[0])):
        raise ValueError(
            f"direction has {dv.shape[0]} dims; the trajectory has "
            f"{traj.get('d_model')}")
    keep = bool(params.get("keep_vectors", False))
    rows = []
    for r in _rows(traj):
        v = np.asarray(r["vector"], dtype=np.float32)
        item = S.coordinate(
            float(v @ dv), S.space_of(r, header=traj), d,
            id=r.get("id"), coords=S.coords_of(r), token=r.get("token"),
            step=r.get("step"), position=r.get("position"),
            n_pooled=r.get("n_pooled"), steps=r.get("steps"))
        if keep:
            item["vector"] = r["vector"]
        rows.append(item)
    from mechbench_compute.lexicon import kinds as K

    header = _header(traj)
    header["projected"] = True
    return K.collection("activations/coordinate", rows, **header)
