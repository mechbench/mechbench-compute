"""The types a lexicon entry is made of.

An op's declaration is the ONE place its parameters are described:
`check_params` refuses what is not declared here, the test suite proves
the declaration equals what the code reads, and the documentation site
renders it. So a parameter's type, default and meaning live beside its
name, not in a table somebody maintains elsewhere.

Writing rule: every sentence here is read by a stranger. Say what a
parameter DOES to the result, in the words a person composing a protocol
would use — never which task introduced it or which experiment wanted
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The reserved root a canonical op is stored under. A protocol never
#: writes it; `lexicon.canonical_path` adds it, `lexicon.resolve`
#: accepts it.
ROOT = "~canonical/ops/"

#: The reserved root a canonical kind is registered under; spelled bare
#: everywhere else, like an op.
KIND_ROOT = "~canonical/kinds/"

#: The one container kind (docs/LEXICON.md §4): `{kind: "collection",
#: item_kind, key, items, ...header}`. Every plural is this.
COLLECTION = "collection"


@dataclass(frozen=True)
class Kind:
    """One data kind, declared for the person who will read it.

    `name` is the bare `family/kind`. `fields` is the record's own
    fields as JSON-Schema property entries (type, description); `required`
    names the ones every instance carries. `extends` names the ancestor
    whose fields this one adds to; refinement is declared, never encoded
    in path depth. `key` is the fields that identify an item when it is
    collected — empty for a kind that is singular by nature (a reduction
    over a set, or one object). `header` is the collection-level fields a
    collection of this kind carries, with one line each. `renderer` and
    `collection_renderer` are the UI bindings, in the registry's shape.
    `platform` marks a kind produced by the platform rather than by an
    op.
    """

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
    platform: bool = False

    @property
    def path(self) -> str:
        return self.name if self.name == COLLECTION else f"{KIND_ROOT}{self.name}"

    @property
    def family(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def collectable(self) -> bool:
        return bool(self.key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "path": self.path, "family": self.family,
            "summary": self.summary, "doc": self.doc,
            "fields": self.fields, "required": list(self.required),
            "extends": self.extends, "key": list(self.key), "header": self.header,
            "renderer": self.renderer, "collection_renderer": self.collection_renderer,
            "platform": self.platform,
        }


@dataclass(frozen=True)
class Emits:
    """What an op produces: a kind, singly or as a collection of it, and
    the prose that says which fields matter."""

    kind: str
    collection: bool = False
    doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "collection": self.collection, "doc": self.doc}


class _Required:
    """Sentinel: the parameter has no default and must be given."""

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()


@dataclass(frozen=True)
class Param:
    """One parameter of an op.

    `type` is a short type expression in the reader's terms — `int`,
    `list[string]`, `object`, `record`, `direction`, `int | "all"` —
    not a Python annotation. `doc` is markdown; its first paragraph is
    the one-line description a table shows, and any further paragraphs
    are the details a page shows beneath it. `default` is the value the
    block uses when the param is absent, as the protocol would write it
    (JSON-shaped), or REQUIRED.
    """

    name: str
    type: str
    doc: str
    default: Any = REQUIRED

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type, "doc": self.doc,
                             "required": self.required}
        if not self.required:
            d["default"] = self.default
        return d


#: The name of a wildcard port: an op declaring one accepts an edge on
#: any port name not otherwise declared, each carrying what the wildcard
#: declares. The names are data — `records/union` stamps them on a
#: coordinate, `direction/add` names its inputs by them.
WILDCARD = "*"


@dataclass(frozen=True)
class Port:
    """One input of an op: what may arrive on a named edge.

    `kind` is a bare kind name; with `many`, the port carries a
    `collection` of that kind, otherwise one object of it. A kind that
    extends the declared one satisfies it. Where two unrelated kinds are
    read the same way, `kind` lists both with ` | `. `required` ports
    must be wired (or given inline) before the node runs.

    A port is filled by an edge from an upstream node, or — for a small
    literal or a stored object — under the node's `inputs` map, as a
    list, an object, or `{"$fetch": …}`. Never under `params`: what a
    node computes on is an input, what it computes with is a param.
    """

    name: str
    kind: str
    doc: str
    required: bool = True
    many: bool = False

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(k.strip() for k in self.kind.split("|"))

    @property
    def wildcard(self) -> bool:
        return self.name == WILDCARD

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "kinds": list(self.kinds),
                "doc": self.doc, "required": self.required, "many": self.many}


@dataclass(frozen=True)
class Op:
    """One canonical operation, described for the person using it.

    `summary`: one sentence — what the op does, in plain words. It is
    what an index and `llms.txt` show, so it has to stand alone.
    `description`: markdown, as long as it needs to be — what the op
    does, the shapes it expects, what the result looks like.
    `inputs`: the typed ports — what arrives by edge, or inline under
    the node's `inputs`. `emits`: the record it produces. `example`: a
    params object that would run, with `$bindings` and `{"$fetch": …}`
    where a value comes from outside the protocol; `example_inputs` the
    node's `inputs` beside it, when the example needs any.
    """

    #: The bare name, `family/op` — what a protocol writes.
    name: str
    summary: str
    description: str
    params: tuple[Param, ...]
    inputs: tuple[Port, ...] = ()
    #: The kind produced, or None for an op whose result is not a bench
    #: object (a tool handler's).
    emits: Emits | None = None
    example: dict[str, Any] | None = None
    example_inputs: dict[str, Any] | None = None

    @property
    def path(self) -> str:
        """The stored identity: `~canonical/ops/<name>`."""
        return f"{ROOT}{self.name}"

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
        """The declared port an edge on `name` lands on: the port of
        that name, else the wildcard, else None."""
        for p in self.inputs:
            if p.name == name:
                return p
        return self.wildcard

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "family": self.family,
            "summary": self.summary,
            "description": self.description,
            "inputs": [p.to_dict() for p in self.inputs],
            "emits": self.emits.to_dict() if self.emits else None,
            "params": [p.to_dict() for p in self.params],
            "example": self.example,
            "example_inputs": self.example_inputs,
        }


def P(name: str, type: str, doc: str, default: Any = REQUIRED) -> Param:
    """Shorthand for a declaration file: `P("top_k", "int", "…", 5)`."""
    return Param(name, type, doc, default)


def In(name: str, kind: str, doc: str, *, required: bool = True, many: bool = False) -> Port:
    """Shorthand for a declaration file: `In("records", "records/record",
    "…", many=True)`."""
    return Port(name, kind, doc, required, many)
