from __future__ import annotations

from typing import Any

from mechbench_compute.support import Architecture, walk_architectures

ARCHITECTURES: tuple[Architecture, ...] = walk_architectures(__name__)

BY_MODEL_TYPE: dict[str, Architecture] = {a.model_type: a for a in ARCHITECTURES}


def for_type(model_type: str | None) -> Architecture | None:
    return BY_MODEL_TYPE.get((model_type or "").lower().removesuffix("_text"))


def for_model(model: Any) -> Architecture:
    model_type = str(getattr(model.config, "model_type", "") or "")
    found = for_type(model_type)
    if found is None:
        raise NotImplementedError(
            f"model_type {model_type!r} is not one the torch backend loads; it loads "
            f"{', '.join(BY_MODEL_TYPE)}")
    return found
