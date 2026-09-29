from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def walk_modules(package: str) -> list[ModuleType]:
    root = importlib.import_module(package)
    found = []
    for info in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        parent_name, _, leaf = info.name.rpartition(".")
        if info.ispkg or leaf.startswith("_"):
            continue
        parent = importlib.import_module(parent_name)
        held = parent.__dict__.get(leaf)
        found.append(importlib.import_module(info.name))
        if held is not None and not isinstance(held, ModuleType):
            setattr(parent, leaf, held)
    return found
