from __future__ import annotations

import importlib
import pkgutil

from mechbench_compute.lexicon._base import Kind


def module_name(kind: str) -> str:
    family, name = kind.split("/", 1)
    return f"{__name__}.{family}.{name.replace('-', '_')}"


def load_platform_kinds() -> tuple[Kind, ...]:
    found: list[Kind] = []
    for info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
        if info.ispkg or info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        kind = getattr(importlib.import_module(info.name), "KIND", None)
        if kind is None:
            raise ImportError(f"{info.name} is under platform/ and declares no KIND")
        if module_name(kind.name) != info.name:
            raise ImportError(
                f"{info.name} declares {kind.name!r}, which belongs at {module_name(kind.name)}: "
                "a platform kind's path is a function of its name")
        found.append(kind)
    return tuple(sorted(found, key=lambda k: k.name))


PLATFORM: tuple[Kind, ...] = load_platform_kinds()
