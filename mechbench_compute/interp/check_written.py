from __future__ import annotations

from mechbench_compute.interp.point_refused import PointRefused


def check_written(model, point: str) -> None:
    architecture = model.architecture
    writes = architecture.writes_of(model.arch)
    if point not in writes:
        why = (architecture.absent_points(model.arch).get(point)
               or f"its layers write {' and '.join(writes)}")
        raise PointRefused(
            "POINT_ABSENT",
            f"`{point}` is not written on this {architecture.name} checkpoint "
            f"({architecture.model_type}): {why}", point)
