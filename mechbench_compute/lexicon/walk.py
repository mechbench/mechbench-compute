from __future__ import annotations

from types import ModuleType

from mechbench_compute.lexicon._base import COLLECTION, Kind
from mechbench_compute.walk_modules import walk_modules


def name_module(package: str, name: str) -> str:
    if "/" not in name:
        return f"{package}.{name}"
    family, leaf = name.split("/", 1)
    return f"{package}.{family}.{leaf.replace('-', '_')}"


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

