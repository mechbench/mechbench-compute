from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import Any

from mechbench_compute.lexicon._base import COLLECTION
from mechbench_compute.lexicon.extension import Extension


@dataclass(frozen=True)
class Manifest:
    ops: tuple[dict[str, Any], ...]
    kinds: tuple[dict[str, Any], ...]
    core: bool = False
    scope: str | None = None

    @cached_property
    def own(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for k in self.kinds:
            out[str(k.get("name"))] = k
            if k.get("path"):
                out[str(k["path"])] = k
        return out

    @cached_property
    def known(self) -> dict[str, dict[str, Any]]:
        return {**read_core_kinds(), **self.own}

    def find_kind(self, name: str) -> dict[str, Any] | None:
        return self.known.get(name)

    def read_ancestry(self, name: str) -> list[str]:
        out: list[str] = []
        cur: str | None = name
        while cur is not None and cur not in out and cur in self.known:
            out.append(cur)
            cur = self.known[cur].get("extends")
        return out

    def read_fields(self, name: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for n in reversed(self.read_ancestry(name)):
            out.update(self.known[n].get("fields") or {})
        return out

    def satisfies(self, actual: str, declared: str) -> bool:
        if declared == COLLECTION:
            return True
        return declared in self.read_ancestry(actual)


@cache
def read_core_kinds() -> dict[str, dict[str, Any]]:
    from mechbench_compute.registry import CORE

    return {k.name: k.to_dict() for k in CORE.kinds()}


@cache
def read_core() -> Manifest:
    from mechbench_compute.registry import CORE

    return Manifest(ops=tuple(m.OP.to_dict() for _, m in sorted(CORE.modules().items())),
                    kinds=tuple(k.to_dict() for k in CORE.kinds()), core=True)


def read_manifest(extension: Extension | dict[str, Any] | Manifest) -> Manifest:
    if isinstance(extension, Manifest):
        return extension
    d = extension.to_dict() if isinstance(extension, Extension) else extension
    provides = d.get("provides") or {}
    owner, project = d.get("owner"), d.get("project")
    return Manifest(ops=tuple(provides.get("ops") or ()), kinds=tuple(provides.get("kinds") or ()),
                    scope=f"{owner}/{project}" if owner and project else None)


def read_kinds_of(spec: Any) -> list[str]:
    return [k.strip() for k in str(spec or "").split("|") if k.strip()]


def read_outputs(op: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = [("output", op["output"])] if op.get("output") else []
    out += [(f"outputs.{port}", o) for port, o in (op.get("outputs") or {}).items()]
    return out
