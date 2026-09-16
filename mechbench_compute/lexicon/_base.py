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

import re
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

#: A stored path may carry a version tail from before the 2026-09
#: renames (`~canonical/ops/factor-cross/1`); the name is what is inside.
_VERSION_TAIL = re.compile(r"/\d+$")


def display_name(name: str) -> str:
    """The bare name inside whatever form arrived: the name itself, or a
    stored path with its root and any version tail."""
    for root in (ROOT, KIND_ROOT):
        if name.startswith(root):
            return _VERSION_TAIL.sub("", name[len(root):])
    return _VERSION_TAIL.sub("", name)


def title(name: str) -> str:
    """The same name, set for reading: `records/cross` -> `Records ::
    Cross`, `activations/capture-attention` -> `Activations :: Capture
    Attention`.

    A rendering, not a second name, which is a distinction this codebase
    has paid for: the composer used to carry a hand-written PascalCase
    label per operation — `FactorCross` for `records/cross` — and it
    drifted out of the lexicon unnoticed until one graph rendered a
    label beside four bare names. So this is MECHANICAL (no table maps
    an operation to a prettier word), TOTAL (a name nobody has declared
    yet renders the same way), and REVERSIBLE (`name_of_title` is the
    inverse, and the tests hold the pair to it).

    Which form goes where is written down on the platform's Names page.
    The short version: a surface a person READS is set this way, and
    anything a person would TYPE — a block string in a graph, a field in
    `ops.json`, provenance — is the name itself."""
    return " :: ".join(
        " ".join(w[:1].upper() + w[1:] for w in segment.split("-"))
        for segment in display_name(name).split("/")
    )


def name_of_title(shown: str) -> str:
    """The name behind a rendered one: `Records :: Cross` ->
    `records/cross`. The inverse of `title`, so the rendering can be
    PROVEN lossless rather than assumed to be."""
    return "/".join(
        segment.lower().replace(" ", "-") for segment in shown.split(" :: ")
    )


@dataclass(frozen=True)
class Metric:
    """A way two items of a kind compare, bound to the kind the way a
    renderer is: `name` is what a protocol writes (`cosine`,
    `jensen-shannon`, `hamming`); `kind` says whether a larger number
    means more alike (`similarity`) or further apart (`distance`);
    `symmetric` is whether m(a, b) = m(b, a) — a tree refuses a metric
    that is not; `options` are the parameters the metric takes, each a
    `Param`. A subtype inherits its ancestor's metrics."""

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
    """One namespace of the two vocabularies: the operations and the kinds
    that share a first segment (`records/`, `direction/`), described for
    the reader who wants to know what the family is FOR before choosing
    a member. `name` is the segment; `summary` is one sentence; `doc` is
    as long as it needs to be. A family with kinds and no operations
    (`model/`, `run/`) is a platform family: its kinds are produced by
    the platform, not by an op."""

    name: str
    summary: str
    doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "summary": self.summary, "doc": self.doc}


@dataclass(frozen=True)
class Value:
    """A value type: a field shape (`space`, `token`, `coords`) or a
    parameter grammar (the position selector, the point vocabulary) that
    many kinds and ops share, declared once so it is described once.
    Not a kind — it is never stored on its own and has no key — but a
    reader meets it on every page that uses it. `fields` are its
    properties in the shape a kind's fields take; `grammar` marks a
    parameter vocabulary rather than a stored shape."""

    name: str
    summary: str
    doc: str = ""
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    grammar: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "summary": self.summary, "doc": self.doc,
                "fields": self.fields, "required": list(self.required),
                "grammar": self.grammar}


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
    `collection_renderer` are the UI bindings, in the registry's shape;
    `metrics` are the comparison bindings (`Metric`), inherited by
    subtypes. `platform` marks a kind produced by the platform rather
    than by an op.
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
    metrics: tuple[Metric, ...] = ()
    platform: bool = False

    @property
    def path(self) -> str:
        return self.name if self.name == COLLECTION else f"{KIND_ROOT}{self.name}"

    @property
    def title(self) -> str:
        """The name set for reading: `Records :: Record`. A rendering,
        derived by rule — see `lexicon.title`."""
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

    `many` and `variadic` are different things, and a port may be
    either, both or neither (task 000397):

    * **`many`** — ONE value that is a collection of the kind. A capture
      node's `records` port takes one collection of records.
    * **`variadic`** — SEVERAL EDGES, each its own value, in a declared
      order. A `zip` node's `branches` port takes one edge per branch,
      and which branch is which is the point. The block receives
      `[{node, value}, …]`, so it knows where each came from.

    A port that is neither takes exactly one edge. Two edges into one
    used to keep whichever came last in the edge list, silently; that is
    refused at load now.
    """

    name: str
    kind: str
    doc: str
    required: bool = True
    many: bool = False
    variadic: bool = False
    #: Bounds on how many edges a variadic port accepts, when it has any.
    min_edges: int | None = None
    max_edges: int | None = None
    #: What this port does when its upstream produced nothing — it failed,
    #: or was itself skipped (task 000399):
    #:
    #: * `fail` (the default) — the run fails, as it always did. A node
    #:   that cannot have this input cannot be trusted to mean anything
    #:   without it.
    #: * `skip` — this node is skipped too, and its own consumers see it
    #:   as missing in turn.
    #: * `placeholder` — the port gets its kind's empty value carrying a
    #:   `missing` marker, and the block decides what to do. Only a port
    #:   that takes a collection can have one: an empty collection is a
    #:   real value, where an empty `direction/vector` is not.
    on_missing: str = "fail"

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(k.strip() for k in self.kind.split("|"))

    @property
    def wildcard(self) -> bool:
        return self.name == WILDCARD

    def arity_error(self, n: int) -> str | None:
        """Why `n` edges is the wrong number for this port, or None."""
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
    def title(self) -> str:
        """The name set for reading: `Records :: Cross`. A rendering,
        derived by rule — see `lexicon.title`."""
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
            "title": self.title,
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


def In(name: str, kind: str, doc: str, *, required: bool = True,
       many: bool = False, variadic: bool = False,
       min_edges: int | None = None, max_edges: int | None = None,
       on_missing: str = "fail") -> Port:
    """Shorthand for a declaration file: `In("records", "records/record",
    "…", many=True)`."""
    return Port(name, kind, doc, required, many, variadic, min_edges,
                max_edges, on_missing)
