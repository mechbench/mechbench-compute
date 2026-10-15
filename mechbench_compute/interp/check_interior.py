from __future__ import annotations


def check_interior(model, point: str) -> None:
    if not model.architecture.supports(point, layer_scoped=True):
        raise ValueError(
            f"the {model.arch.model_type!r} forward on the {model.architecture.backend} "
            f"backend does not read `{point}`")
