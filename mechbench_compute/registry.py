from __future__ import annotations

import dataclasses
import importlib.metadata
import json
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from mechbench_compute.lexicon._base import COLLECTION, Kind, Op, Output
from mechbench_compute.lexicon.address import (
    OP_ROOT,
    AddressError,
    address_kind,
    address_op,
    is_extension_spelling,
    parse_op,
)
from mechbench_compute.lexicon.extension import Extension, hash_extension
from mechbench_compute.lexicon.walk import walk_kinds, walk_ops

GROUP = "mechbench.extensions"

SEALED = frozenset({"run", "sandbox", "provider", "model", "policy", "extension"})


class RestartRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class Resolved:
    op: Op
    tier: str
    module: ModuleType
    source: str
    name: str
    version: int | None = None

    @property
    def digest(self) -> str | None:
        return None if self.tier == "core" else self.source.rsplit("@", 1)[1]

    @property
    def pinned(self) -> str:
        return f"{OP_ROOT}{self.name}" if self.tier == "core" else f"{self.name}@{self.digest}"


@dataclass
class Table:
    ops: dict[str, Resolved] = field(default_factory=dict)
    kinds: dict[str, Kind] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    sorted_ops: tuple[Op, ...] = ()
    sorted_kinds: tuple[Kind, ...] = ()


class CoreSource:
    tier = "core"

    def __init__(self, ops_package: str = "mechbench_compute.ops",
                 kinds_package: str = "mechbench_compute.lexicon.kinds") -> None:
        self.ops_package = ops_package
        self.kinds_package = kinds_package
        self._modules: dict[str, ModuleType] | None = None
        self._kinds: tuple[Kind, ...] | None = None

    def kinds(self) -> tuple[Kind, ...]:
        if self._kinds is None:
            self._kinds = walk_kinds(self.kinds_package)
        return self._kinds

    def modules(self) -> dict[str, ModuleType]:
        if self._modules is None:
            self._modules = walk_ops(self.ops_package)
        return self._modules

    def load(self, table: Table) -> None:
        for kind in self.kinds():
            table.kinds[kind.name] = kind
        for name, mod in self.modules().items():
            table.ops[name] = Resolved(mod.OP, self.tier, mod, "core", name)


def qualify(kind: str, scope: str, own: frozenset[str]) -> str:
    return " | ".join(address_kind(scope, k) if k in own else k
                      for k in (s.strip() for s in kind.split("|")))


def qualify_output(out: Output, scope: str, own: frozenset[str]) -> Output:
    return dataclasses.replace(
        out, kind=qualify(out.kind, scope, own),
        otherwise=tuple(dataclasses.replace(o, kind=qualify(o.kind, scope, own)) for o in out.otherwise))


def qualify_op(op: Op, scope: str, own: frozenset[str]) -> Op:
    return dataclasses.replace(
        op, name=address_op(scope, op.name),
        inputs=tuple(dataclasses.replace(p, kind=qualify(p.kind, scope, own)) for p in op.inputs),
        output=qualify_output(op.output, scope, own) if op.output else None,
        outputs=({k: qualify_output(v, scope, own) for k, v in op.outputs.items()}
                 if op.outputs is not None else None))


def qualify_kind(kind: Kind, scope: str, own: frozenset[str]) -> Kind:
    extends = kind.extends
    return dataclasses.replace(
        kind, name=address_kind(scope, kind.name),
        extends=address_kind(scope, extends) if extends in own else extends)


@dataclass(frozen=True)
class Loaded:
    extension: Extension
    digest: str
    dist_version: str | None
    ops: dict[str, Resolved]
    kinds: dict[str, Kind]


def read_installed() -> list[dict[str, Any]]:
    path = Path.home() / ".mechbench" / "extensions" / "installed.json"
    if not path.is_file():
        return []
    held = json.loads(path.read_text())
    return list(held.values() if isinstance(held, dict) else held)


class InstalledSource:
    tier = "installed"

    def __init__(self, find: Callable[[], Iterable[Any]] | None = None,
                 installed: Callable[[], list[dict[str, Any]]] = read_installed) -> None:
        self.find = find or (lambda: importlib.metadata.entry_points(group=GROUP))
        self.installed = installed
        self.loaded: dict[str, Loaded] = {}
        self.dists: dict[str, str | None] = {}
        self.refused: dict[str, str] = {}
        self.restart: list[str] = []

    def discover(self, reserved: frozenset[str], core_kinds: Mapping[str, Kind]) -> None:
        self.refused = {}
        self.restart = []
        for ep in self.find():
            dist = getattr(getattr(ep, "dist", None), "version", None)
            if ep.name in self.dists and dist is not None and self.dists[ep.name] != dist:
                self.restart.append(
                    f"extension {ep.name!r} changed from {self.dists[ep.name]} to {dist} "
                    "after it was loaded; Python cannot unload it, so the runner must restart")
                continue
            scope = ep.name
            try:
                manifest = ep.load()
                if not isinstance(manifest, Extension):
                    raise ValueError(f"{ep.value} is not an Extension")
                scope = manifest.name
                held = self.loaded.get(manifest.address)
                loaded = self.load_one(manifest, dist, reserved, core_kinds)
                if held is not None and held.digest != loaded.digest:
                    self.restart.append(
                        f"{manifest.address} changed from @{held.extension.version} ({held.digest}) to "
                        f"@{manifest.version} ({loaded.digest}) after it was loaded; Python cannot "
                        "unload it, so the runner must restart")
                    continue
                clash = next((o for o in loaded.ops for other in self.loaded.values()
                              if other.extension.address != manifest.address and o in other.ops), None)
                if clash is not None:
                    raise ValueError(f"{clash} is already provided by another extension of {manifest.name}")
            except Exception as e:  # noqa: BLE001
                self.refused[scope] = f"extension {ep.name!r} ({ep.value}) was refused at load: {e}"
                continue
            self.loaded[manifest.address] = loaded
            self.dists[ep.name] = dist

    def load_one(self, ext: Extension, dist: str | None, reserved: frozenset[str],
                 core_kinds: Mapping[str, Kind]) -> Loaded:
        if ext.owner in reserved:
            raise ValueError(f"owner {ext.owner!r} is a core family name, which is reserved")
        scope = ext.name
        declared = ext.kinds
        own = frozenset(k.name for k in declared)
        kinds: dict[str, Kind] = {}
        for k in declared:
            if k.family in SEALED:
                raise ValueError(f"kind {k.name!r}: the {k.family!r} family is sealed")
            if not k.speak:
                raise ValueError(f"kind {k.name!r} declares no speak; every extension kind says itself")
            if k.extends is None or not (k.extends in own or k.extends in core_kinds):
                raise ValueError(f"kind {k.name!r} extends {k.extends!r}, which is neither a core "
                                 "kind nor one this extension declares")
            q = qualify_kind(k, scope, own)
            kinds[q.name] = q
        recorded = [r.get("hash") for r in self.installed()
                    if r.get("address") == ext.address and r.get("version") == ext.version]
        digest = recorded[0] if recorded and recorded[0] else hash_extension(ext.to_dict())
        source = f"{ext.address}@{digest}"
        known = set(core_kinds) | set(kinds)
        ops: dict[str, Resolved] = {}
        for name, mod in ext.modules().items():
            op = qualify_op(mod.OP, scope, own)
            ports = [k for p in op.inputs for k in p.kinds if not p.wildcard]
            outs = [op.output] if op.output else list((op.outputs or {}).values())
            unknown = sorted({k for k in [*ports, *(o.kind for o in outs)]
                              if k not in known and k != COLLECTION})
            if unknown:
                raise ValueError(f"{op.name} names kinds nobody declares: {unknown}")
            ops[op.name] = Resolved(op, self.tier, mod, source, op.name, ext.version)
        return Loaded(ext, digest, dist, ops, kinds)

    def load(self, table: Table, reserved: frozenset[str]) -> None:
        self.discover(reserved, table.kinds)
        for loaded in self.loaded.values():
            table.kinds.update(loaded.kinds)
            table.ops.update(loaded.ops)
        table.refused.update(self.refused)


class Registry:
    def __init__(self, sources: Iterable[Any]) -> None:
        self.sources = tuple(sources)
        self.generation = 0
        self._table: Table | None = None
        self._partial: Table | None = None

    def table(self) -> Table:
        if self._table is not None:
            return self._table
        if self._partial is not None:
            return self._partial
        self._partial = Table()
        try:
            self._table = self.build(self._partial)
        finally:
            self._partial = None
        return self._table

    def build(self, table: Table) -> Table:
        for source in self.sources:
            if source.tier == "core":
                for kind in source.kinds():
                    table.kinds[kind.name] = kind
        for source in self.sources:
            if source.tier == "core":
                source.load(table)
        reserved = frozenset(k.split("/", 1)[0] for k in (*table.ops, *table.kinds)) | SEALED
        for source in self.sources:
            if source.tier != "core":
                source.load(table, reserved)
        table.sorted_ops = tuple(r.op for _, r in sorted(table.ops.items()))
        table.sorted_kinds = tuple(sorted(table.kinds.values(),
                                          key=lambda k: (k.name == COLLECTION, k.name)))
        return table

    def refresh(self) -> None:
        self._table = None
        self.generation += 1
        self.table()
        restart = [m for s in self.sources for m in getattr(s, "restart", ())]
        if restart:
            raise RestartRequired("; ".join(restart))

    def find(self, spelling: str) -> Resolved | None:
        try:
            return self.resolve(spelling)
        except KeyError:
            return None

    def resolve(self, spelling: str) -> Resolved:
        ops = self.table().ops
        s = str(spelling).strip()
        hit = ops.get(s)
        if hit is not None:
            return hit
        if not is_extension_spelling(s):
            hit = ops.get(s[len(OP_ROOT):]) if s.startswith(OP_ROOT) else None
            if hit is None or hit.tier != "core":
                raise KeyError(self.explain(spelling))
            return hit
        try:
            address = parse_op(s)
        except AddressError as e:
            raise KeyError(f"unknown block {spelling!r}: {e}") from None
        hit = ops.get(address.bare)
        if hit is None:
            raise KeyError(self.explain(spelling))
        if address.version is not None and address.version != hit.version:
            raise KeyError(f"unknown block {spelling!r}: the installed {hit.source.split('@', 1)[0]} "
                           f"is @{hit.version}, not @{address.version}")
        if address.pin is not None and address.pin != hit.digest:
            raise KeyError(f"unknown block {spelling!r}: pinned to {address.pin}, but the installed "
                           f"{hit.source.split('@', 1)[0]}@{hit.version} is {hit.digest}")
        return hit

    def explain(self, spelling: str) -> str:
        if not is_extension_spelling(str(spelling)):
            from mechbench_compute.lexicon import explain_core

            return explain_core(spelling)
        try:
            address = parse_op(str(spelling))
        except AddressError as e:
            return f"unknown block {spelling!r}: {e}"
        if address.bare in self.table().ops:
            try:
                self.resolve(spelling)
            except KeyError as e:
                return str(e.args[0])
        refused = self.table().refused.get(str(address.scope))
        if refused is not None:
            return f"unknown block {spelling!r}: {refused}"
        return (f"unknown block {spelling!r}: no installed extension provides {address.bare}; "
                "a runner installs an extension when a claimed job needs it")

    def kind(self, name: str) -> Kind:
        return self.table().kinds[name]

    def ops(self) -> tuple[Op, ...]:
        return self.table().sorted_ops

    def kinds(self) -> tuple[Kind, ...]:
        return self.table().sorted_kinds

    def resolved(self) -> tuple[Resolved, ...]:
        return tuple(r for _, r in sorted(self.table().ops.items()))


CORE = CoreSource()

INSTALLED = InstalledSource()

REGISTRY = Registry((CORE, INSTALLED))
