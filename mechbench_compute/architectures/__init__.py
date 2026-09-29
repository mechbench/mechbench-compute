from __future__ import annotations

from mechbench_compute.support import Architecture
from mechbench_compute.walk_modules import walk_modules


def walk_architectures(package: str) -> tuple[Architecture, ...]:
    found = []
    for mod in walk_modules(package):
        arch = getattr(mod, "ARCH", None)
        if arch is None:
            raise ImportError(f"{mod.__name__} is under architectures/ and declares no ARCH")
        if f"{package}.{arch.model_type}" != mod.__name__:
            raise ImportError(
                f"{mod.__name__} declares {arch.model_type!r}, which belongs at "
                f"{package}.{arch.model_type}: an architecture's path is its model_type")
        found.append(arch)
    return tuple(sorted(found, key=lambda a: a.model_type))


ARCHITECTURES: tuple[Architecture, ...] = walk_architectures(__name__)

BY_MODEL_TYPE: dict[str, Architecture] = {a.model_type: a for a in ARCHITECTURES}


def for_type(model_type: str | None) -> Architecture | None:
    return BY_MODEL_TYPE.get((model_type or "").lower())


def read_model_type(model) -> str:
    if hasattr(model, "args") and not hasattr(model, "language_model"):
        return str(getattr(model.args, "model_type", "") or "").lower()
    config = model.config
    text = getattr(config, "text_config", None)
    raw = getattr(text, "model_type", None) or getattr(config, "model_type", "") or ""
    return str(raw).lower().removesuffix("_text")


def for_model(model) -> Architecture:
    model_type = read_model_type(model)
    found = for_type(model_type)
    if found is None:
        raise NotImplementedError(
            f"model_type {model_type!r} is not one compute loads; it loads "
            f"{', '.join(BY_MODEL_TYPE)}")
    return found
