from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ROOT = "~canonical/ops/"

KIND_ROOT = "~canonical/kinds/"

COLLECTION = "collection"

_VERSION_TAIL = re.compile(r"/\d+$")


def display_name(name: str) -> str:
    for root in (ROOT, KIND_ROOT):
        if name.startswith(root):
            return _VERSION_TAIL.sub("", name[len(root):])
    return _VERSION_TAIL.sub("", name)


def title(name: str) -> str:
    return " :: ".join(
        " ".join(w[:1].upper() + w[1:] for w in segment.split("-"))
        for segment in display_name(name).split("/")
    )


def name_of_title(shown: str) -> str:
    return "/".join(
        segment.lower().replace(" ", "-") for segment in shown.split(" :: ")
    )


@dataclass(frozen=True)
class Metric:
    name: str
    kind: str
    symmetric: bool
    doc: str
    options: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "symmetric": self.symmetric,
                "doc": self.doc, "options": [o.to_dict() for o in self.options]}


@dataclass(frozen=True)
class Family:
    name: str
    summary: str
    doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "summary": self.summary, "doc": self.doc}


@dataclass(frozen=True)
class Value:
    name: str
    summary: str
    doc: str = ""
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    grammar: bool = False
    choices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "summary": self.summary, "doc": self.doc,
             "fields": self.fields, "required": list(self.required),
             "grammar": self.grammar}
        if self.choices:
            d["choices"] = list(self.choices)
        return d


@dataclass(frozen=True)
class Kind:
    name: str
    summary: str
    doc: str = ""
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    extends: str | None = None
    key: tuple[str, ...] = ()
    header: dict[str, str] = field(default_factory=dict)
    renderer: dict[str, Any] | None = None
    collection_renderer: dict[str, Any] | None = None
    metrics: tuple[Metric, ...] = ()
    platform: bool = False

    @property
    def path(self) -> str:
        return self.name if self.name == COLLECTION else f"{KIND_ROOT}{self.name}"

    @property
    def title(self) -> str:
        return title(self.name)

    @property
    def family(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def collectable(self) -> bool:
        return bool(self.key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "path": self.path, "title": self.title,
            "family": self.family,
            "summary": self.summary, "doc": self.doc,
            "fields": self.fields, "required": list(self.required),
            "extends": self.extends, "key": list(self.key), "header": self.header,
            "renderer": self.renderer, "collection_renderer": self.collection_renderer,
            "metrics": [m.to_dict() for m in self.metrics],
            "platform": self.platform,
        }


@dataclass(frozen=True)
class Otherwise:
    kind: str
    collection: bool = False
    param: str | None = None
    equals: Any = None
    port: str | None = None

    def to_dict(self) -> dict[str, Any]:
        when: dict[str, Any] = (
            {"port": self.port} if self.port is not None else {"param": self.param, "equals": self.equals})
        return {"kind": self.kind, "collection": self.collection, "when": when}


@dataclass(frozen=True)
class Output:
    kind: str
    collection: bool = False
    doc: str = ""
    otherwise: tuple[Otherwise, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "collection": self.collection, "doc": self.doc}
        if self.otherwise:
            d["otherwise"] = [o.to_dict() for o in self.otherwise]
        return d


Emits = Output


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()


TYPE_WORDS = frozenset({"string", "int", "float", "bool", "null",
                        "selector", "model", "object", "json", "callable",
                        "ref"})


@dataclass(frozen=True)
class TypeNode:
    form: str
    word: str = ""
    of: tuple[TypeNode, ...] = ()


def _split_top(text: str, sep: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def parse_type(text: str) -> tuple[TypeNode, ...]:
    alts: list[TypeNode] = []
    for raw in _split_top(text, "|"):
        alt = raw.strip()
        if len(alt) >= 2 and alt[0] == alt[-1] == '"':
            alts.append(TypeNode("literal", alt[1:-1]))
        elif alt.startswith("list[") and alt.endswith("]"):
            alts.append(TypeNode("list", of=parse_type(alt[5:-1])))
        elif alt.startswith("map[") and alt.endswith("]"):
            key, _, rest = alt[4:-1].partition(",")
            if key.strip() != "string" or not rest.strip():
                raise ValueError(f"{alt!r}: a map is `map[string, T]`")
            alts.append(TypeNode("map", of=parse_type(rest)))
        elif alt in TYPE_WORDS:
            alts.append(TypeNode("word", alt))
        else:
            raise ValueError(f"{alt!r} in {text!r} is not in the type grammar")
    return tuple(alts)


def type_words(text: str) -> frozenset[str]:
    def walk(nodes: tuple[TypeNode, ...]) -> set[str]:
        out: set[str] = set()
        for n in nodes:
            if n.form == "word":
                out.add(n.word)
            out |= walk(n.of)
        return out
    return frozenset(walk(parse_type(text)))


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    doc: str
    default: Any = REQUIRED
    choices: tuple[str, ...] = ()
    value: str | None = None
    fields: tuple[Param, ...] = ()
    stored: str | None = None
    reference: bool = False

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type, "doc": self.doc,
                             "required": self.required}
        if not self.required:
            d["default"] = self.default
        if self.choices:
            d["choices"] = list(self.choices)
        if self.value:
            d["value"] = self.value
        if self.fields:
            d["fields"] = [f.to_dict() for f in self.fields]
        if self.stored:
            d["stored"] = self.stored
        if self.reference:
            d["reference"] = True
        return d


WILDCARD = "*"


@dataclass(frozen=True)
class Port:
    name: str
    kind: str
    doc: str
    required: bool = True
    many: bool = False
    variadic: bool = False
    min_edges: int | None = None
    max_edges: int | None = None
    on_missing: str = "fail"

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(k.strip() for k in self.kind.split("|"))

    @property
    def wildcard(self) -> bool:
        return self.name == WILDCARD

    def arity_error(self, n: int) -> str | None:
        if not self.variadic:
            if n > 1:
                return (f"takes one edge; {n} arrive. Either the op's port "
                        f"should be variadic, or one of these edges belongs "
                        f"elsewhere — before this was refused, the last one "
                        f"in the list silently won")
            return None
        if self.min_edges is not None and n < self.min_edges:
            return f"takes at least {self.min_edges} edges; {n} arrive"
        if self.max_edges is not None and n > self.max_edges:
            return f"takes at most {self.max_edges} edges; {n} arrive"
        return None

    def to_dict(self) -> dict[str, Any]:
        out = {"name": self.name, "kind": self.kind, "kinds": list(self.kinds),
               "doc": self.doc, "required": self.required, "many": self.many,
               "variadic": self.variadic, "on_missing": self.on_missing}
        if self.min_edges is not None:
            out["min_edges"] = self.min_edges
        if self.max_edges is not None:
            out["max_edges"] = self.max_edges
        return out


@dataclass(frozen=True)
class Op:
    name: str
    summary: str
    description: str
    params: tuple[Param, ...]
    inputs: tuple[Port, ...] = ()
    output: Output | None = None
    requires: str = "pure"
    example: dict[str, Any] | None = None
    example_inputs: dict[str, Any] | None = None

    @property
    def path(self) -> str:
        return f"{ROOT}{self.name}"

    @property
    def title(self) -> str:
        return title(self.name)

    @property
    def family(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def param_names(self) -> frozenset[str]:
        return frozenset(p.name for p in self.params)

    @property
    def port_names(self) -> frozenset[str]:
        return frozenset(p.name for p in self.inputs)

    @property
    def wildcard(self) -> Port | None:
        return next((p for p in self.inputs if p.wildcard), None)

    def port(self, name: str) -> Port | None:
        for p in self.inputs:
            if p.name == name:
                return p
        return self.wildcard

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "title": self.title,
            "family": self.family,
            "requires": self.requires,
            "summary": self.summary,
            "description": self.description,
            "inputs": [p.to_dict() for p in self.inputs],
            "output": self.output.to_dict() if self.output else None,
            "params": [p.to_dict() for p in self.params],
            "example": self.example,
            "example_inputs": self.example_inputs,
        }


def P(name: str, type: str, doc: str, default: Any = REQUIRED, *,
      choices: tuple[str, ...] = (), value: str | None = None,
      fields: tuple[Param, ...] = (), stored: str | None = None,
      reference: bool = False) -> Param:
    return Param(name, type, doc, default, choices, value, fields, stored, reference)


def In(name: str, kind: str, doc: str, *, required: bool = True,
       many: bool = False, variadic: bool = False,
       min_edges: int | None = None, max_edges: int | None = None,
       on_missing: str = "fail") -> Port:
    return Port(name, kind, doc, required, many, variadic, min_edges,
                max_edges, on_missing)
