from __future__ import annotations

from mechbench_compute.dialects import ToolDialect
from mechbench_compute.walk_modules import walk_modules


def walk_dialects(package: str) -> tuple[ToolDialect, ...]:
    found = []
    for mod in walk_modules(package):
        dialect = getattr(mod, "DIALECT", None)
        if dialect is None:
            raise ImportError(f"{mod.__name__} is under tool_dialects/ and declares no DIALECT")
        found.append(dialect)
    return tuple(found)


DIALECTS: tuple[ToolDialect, ...] = walk_dialects(__name__)
