from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any

from mechbench_compute.conformance.check_ops import find_idioms
from mechbench_compute.conformance.finding import Finding
from mechbench_compute.conformance.manifest import Manifest, read_core_kinds
from mechbench_compute.lexicon._base import COLLECTION
from mechbench_compute.registry import SEALED

KIND_NAME = re.compile(r"[a-z0-9-]+/[a-z0-9-]+")


def check_kind(kind: Mapping[str, Any], m: Manifest) -> Iterator[Finding]:
    name = str(kind.get("name") or "")
    if name != COLLECTION and not KIND_NAME.fullmatch(name):
        yield Finding("KIND_NAME_INVALID", name, "a kind's name is two levels, family/leaf")
    s = str(kind.get("summary") or "").strip()
    if not s or s[-1] not in ".?!":
        yield Finding("KIND_NOT_DESCRIBED", name, "a kind's summary is one sentence")
    fields = dict(kind.get("fields") or {})
    texts = [("summary", str(kind.get("summary") or "")), ("doc", str(kind.get("doc") or ""))]
    texts += [(f"field {f}", str(spec.get("description") or "")) for f, spec in fields.items()]
    texts += [(f"header {h}", str(v)) for h, v in (kind.get("header") or {}).items()]
    yield from find_idioms(name, texts)
    for f, spec in fields.items():
        if not (spec.get("type") and spec.get("description")):
            yield Finding("KIND_FIELD_UNDESCRIBED", f"{name}.{f}", "a field has a type and a description")
    extends = kind.get("extends")
    if extends and m.find_kind(str(extends)) is None:
        yield Finding("KIND_EXTENDS_UNKNOWN", name, f"extends unknown {extends!r}")
    inherited = {**m.read_fields(str(extends)), **fields} if extends else fields
    for r in kind.get("required") or ():
        if r not in inherited:
            yield Finding("KIND_REQUIRED_UNKNOWN", name, f"required {r!r} is not a field of it or an ancestor")
    try:
        json.dumps(kind)
    except (TypeError, ValueError) as e:
        yield Finding("NOT_JSON", name, f"the declaration has no JSON form: {e}")
    if not m.core:
        yield from check_extension_kind(kind, name, m)


def check_extension_kind(kind: Mapping[str, Any], name: str, m: Manifest) -> Iterator[Finding]:
    core = read_core_kinds()
    if not kind.get("speak"):
        yield Finding("NO_SPEAK", name, "declares no speak; every extension kind says itself")
    if name in core:
        yield Finding("KIND_SHADOWS_CORE", name, "a core kind's name; a short name inside an extension "
                      "reaches core first")
    if name.split("/", 1)[0] in SEALED:
        yield Finding("KIND_SEALED_FAMILY", name, f"the {name.split('/', 1)[0]!r} family is sealed")
    extends = kind.get("extends")
    if not extends:
        yield Finding("KIND_NO_EXTENDS", name, "an extension kind extends a core kind or one of its own")
        return
    ancestors = [a for a in m.read_ancestry(str(extends)) if a in core and a not in m.own]
    inherited: dict[str, Any] = {}
    header: dict[str, Any] = {}
    for a in reversed(ancestors):
        inherited.update(core[a].get("fields") or {})
        header.update(core[a].get("header") or {})
    for f, spec in (kind.get("fields") or {}).items():
        if f in inherited and inherited[f] != spec:
            yield Finding("KIND_CHANGES_ANCESTOR", f"{name}.{f}", f"changes the field {f!r} a core ancestor declares")
    for h, doc in (kind.get("header") or {}).items():
        if h in header and header[h] != doc:
            yield Finding("KIND_CHANGES_ANCESTOR", name, f"changes the header field {h!r} a core ancestor declares")


def check_lattice(m: Manifest) -> Iterator[Finding]:
    for kind in m.kinds:
        name = str(kind.get("name") or "")
        seen: list[str] = []
        cur = m.find_kind(name)
        while cur is not None and cur.get("extends"):
            nxt = str(cur["extends"])
            if nxt in seen or nxt == name:
                yield Finding("KIND_CYCLE", name, f"the lattice cycles at {nxt}")
                break
            seen.append(nxt)
            cur = m.find_kind(nxt)


def check_kinds(m: Manifest) -> Iterator[Finding]:
    names = [str(k.get("name") or "") for k in m.kinds]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        yield Finding("KIND_DUPLICATE", dup, "two kinds share this name")
    for kind in m.kinds:
        yield from check_kind(kind, m)
    yield from check_lattice(m)
