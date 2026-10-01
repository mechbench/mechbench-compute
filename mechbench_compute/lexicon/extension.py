from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

from mechbench_compute.lexicon._base import Kind, Op
from mechbench_compute.lexicon.address import (
    AddressError,
    address_extension,
    address_kind,
    address_op,
    parse_extension,
)
from mechbench_compute.lexicon.walk import walk_kinds, walk_ops

TIERS = ("installed",)

PIN_FIELDS = ("owner", "project", "name", "version", "tier", "provides", "needs", "min_compute", "package", "links")

PROVIDES_DEFAULTS = {"ops": [], "kinds": [], "marks": [], "architectures": []}
OP_DEFAULTS = {"inputs": [], "output": None, "params": [], "needs": [],
               "resume": {"level": "reproducible", "items": False}, "deterministic": True}
PORT_DEFAULTS = {"doc": "", "required": True, "many": False, "variadic": False, "on_missing": "fail"}
KIND_DEFAULTS = {"doc": "", "fields": {}, "required": [], "extends": None, "key": [], "header": {},
                 "metrics": [], "platform": False, "version": 1}


@dataclass(frozen=True)
class Package:
    name: str
    python: str = ">=3.12"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "python": self.python}


@dataclass(frozen=True)
class Extension:
    name: str
    extension: str
    version: int
    module: str
    package: Package
    min_compute: str
    tier: str = "installed"
    links: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parts = self.name.split("/")
        if len(parts) != 2:
            raise ValueError(f"extension name {self.name!r} is `<owner>/<project>`")
        try:
            parse_extension(f"{self.address}@{self.version}")
        except AddressError as e:
            raise ValueError(f"extension {self.name!r}: {e}") from None
        if self.tier not in TIERS:
            raise ValueError(f"{self.address}: tier {self.tier!r} is not one of {TIERS}")

    @property
    def owner(self) -> str:
        return self.name.split("/")[0]

    @property
    def project(self) -> str:
        return self.name.split("/")[1]

    @property
    def address(self) -> str:
        return address_extension(self.name, self.extension)

    def modules(self) -> dict[str, ModuleType]:
        return walk_extension_ops(self.module)

    @property
    def ops(self) -> tuple[Op, ...]:
        return tuple(m.OP for _, m in sorted(self.modules().items()))

    @property
    def kinds(self) -> tuple[Kind, ...]:
        return walk_extension_kinds(self.module)

    @property
    def marks(self) -> tuple[dict[str, Any], ...]:
        return read_marks(self.module)

    @property
    def needs(self) -> frozenset[str]:
        return frozenset().union(*(op.needs for op in self.ops))

    def to_dict(self) -> dict[str, Any]:
        mods = self.modules()
        return {
            "kind": "extension",
            "owner": self.owner,
            "project": self.project,
            "name": self.extension,
            "version": self.version,
            "tier": self.tier,
            "provides": {
                "ops": [{**mods[n].OP.to_dict(), "path": address_op(self.name, n), "entry": mods[n].__name__}
                        for n in sorted(mods)],
                "kinds": [{**k.to_dict(), "path": address_kind(self.name, k.name)} for k in self.kinds],
                "marks": list(self.marks),
                "architectures": [],
            },
            "needs": sorted(self.needs),
            "min_compute": self.min_compute,
            "package": self.package.to_dict(),
            "conformance": None,
            "state": "draft",
            "visibility": "private",
            "party": "third",
            "links": dict(self.links),
            "flags": [],
            "promoted": {},
            "provenance": dict(self.provenance),
        }


def import_subpackage(module: str, sub: str) -> str | None:
    name = f"{module}.{sub}"
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:
            return None
        raise
    return name


@cache
def walk_extension_ops(module: str) -> dict[str, ModuleType]:
    package = import_subpackage(module, "ops")
    return walk_ops(package) if package else {}


@cache
def walk_extension_kinds(module: str) -> tuple[Kind, ...]:
    package = import_subpackage(module, "kinds")
    return walk_kinds(package) if package else ()


@cache
def read_marks(module: str) -> tuple[dict[str, Any], ...]:
    root = Path(importlib.import_module(module).__file__).parent / "marks"
    if not root.is_dir():
        return ()
    return tuple(json.loads(p.read_text()) for p in sorted(root.glob("*.json")))


def format_number(x: float) -> str:
    if not math.isfinite(x):
        raise ValueError(f"{x!r} has no JSON form")
    if x == 0:
        return "0"
    if x.is_integer() and abs(x) < 1e21:
        return str(int(x))
    sign, digits, exp = Decimal(repr(x)).normalize().as_tuple()
    s = "".join(map(str, digits))
    k = len(s)
    n = exp + k
    if k <= n <= 21:
        out = s + "0" * (n - k)
    elif 0 < n <= 21:
        out = f"{s[:n]}.{s[n:]}"
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + s
    else:
        e = n - 1
        mant = s[0] + (f".{s[1:]}" if k > 1 else "")
        out = f"{mant}e{'+' if e >= 0 else '-'}{abs(e)}"
    return ("-" if sign else "") + out


def encode_canonical(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(encode_canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda k: str(k).encode("utf-16-be"))
        return "{" + ",".join(f"{json.dumps(str(k), ensure_ascii=False)}:{encode_canonical(value[k])}"
                              for k in keys) + "}"
    raise TypeError(f"{type(value).__name__} has no JSON form")


def fill(value: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    return {**copy.deepcopy(defaults), **value}


def fill_param(param: dict[str, Any]) -> dict[str, Any]:
    out = fill(param, {"doc": ""})
    if "fields" in out:
        out["fields"] = [fill_param(f) for f in out["fields"]]
    return out


def fill_output(output: dict[str, Any]) -> dict[str, Any]:
    out = fill(output, {"collection": False})
    if "otherwise" in out:
        out["otherwise"] = [fill(o, {"collection": False}) for o in out["otherwise"]]
    return out


def fill_op(op: dict[str, Any]) -> dict[str, Any]:
    out = fill(op, OP_DEFAULTS)
    out["inputs"] = [fill(p, PORT_DEFAULTS) for p in out["inputs"]]
    out["params"] = [fill_param(p) for p in out["params"]]
    out["resume"] = fill(out["resume"], {"items": False})
    if out["output"] is not None:
        out["output"] = fill_output(out["output"])
    if out.get("outputs") is not None:
        out["outputs"] = {k: fill_output(v) for k, v in out["outputs"].items()}
    return out


def fill_kind(kind: dict[str, Any]) -> dict[str, Any]:
    out = fill(kind, KIND_DEFAULTS)
    out["metrics"] = [fill(m, {"doc": ""}) for m in out["metrics"]]
    return out


def declare(manifest: dict[str, Any]) -> dict[str, Any]:
    out = {k: copy.deepcopy(manifest[k]) for k in PIN_FIELDS if k in manifest}
    out.setdefault("needs", [])
    out.setdefault("links", {})
    provides = fill(out.get("provides") or {}, PROVIDES_DEFAULTS)
    provides["ops"] = [fill_op(o) for o in provides["ops"]]
    provides["kinds"] = [fill_kind(k) for k in provides["kinds"]]
    out["provides"] = provides
    for k in ("owner", "project"):
        if isinstance(out.get(k), str):
            out[k] = out[k].lower()
    return out


def hash_extension(manifest: dict[str, Any]) -> str:
    raw = encode_canonical(declare(manifest)).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()
