from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from mechbench_compute.lexicon._base import COLLECTION, Kind


def name_module(package: str, name: str) -> str:
    if "/" not in name:
        return f"{package}.{name}"
    family, leaf = name.split("/", 1)
    return f"{package}.{family}.{leaf.replace('-', '_')}"


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


def walk_ops(package: str) -> dict[str, ModuleType]:
    found: dict[str, ModuleType] = {}
    for mod in walk_modules(package):
        op = getattr(mod, "OP", None)
        if op is None:
            raise ImportError(f"{mod.__name__} is under ops/ and declares no OP")
        if name_module(package, op.name) != mod.__name__:
            raise ImportError(
                f"{mod.__name__} declares {op.name!r}, which belongs at "
                f"{name_module(package, op.name)}: an operation's path is a function of its name")
        found[op.name] = mod
    return found


def walk_kinds(package: str) -> tuple[Kind, ...]:
    found: list[Kind] = []
    for mod in walk_modules(package):
        kind = getattr(mod, "KIND", None)
        if kind is None:
            raise ImportError(f"{mod.__name__} is under kinds/ and declares no KIND")
        if name_module(package, kind.name) != mod.__name__:
            raise ImportError(
                f"{mod.__name__} declares {kind.name!r}, which belongs at {name_module(package, kind.name)}: "
                "a kind's path is a function of its name")
        found.append(kind)
    return tuple(sorted(found, key=lambda k: (k.name == COLLECTION, k.name)))
