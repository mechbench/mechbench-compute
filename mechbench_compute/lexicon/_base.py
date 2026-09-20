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
    #: A vocabulary grammar's words, when it is one: the point names. An
    #: editor offers them; a param may narrow them with its own `choices`.
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
class Otherwise:
    """What an op produces in place of its usual kind, and the one thing
    about the node that decides it: a param's value, or an input port
    being filled. `trajectory/aggregate` with `as: "vectors"` produces
    `activations/vector`, not `trajectory/summary`; a composer that knew
    only the usual kind would refuse to wire it into `direction/fit`,
    which is exactly where it goes."""

    kind: str
    collection: bool = False
    #: The param whose value decides it, and the value.
    param: str | None = None
    equals: Any = None
    #: The input port whose being filled decides it.
    port: str | None = None

    def to_dict(self) -> dict[str, Any]:
        when: dict[str, Any] = (
            {"port": self.port} if self.port is not None else {"param": self.param, "equals": self.equals})
        return {"kind": self.kind, "collection": self.collection, "when": when}


@dataclass(frozen=True)
class Output:
    """What an op produces — its one output: a kind, singly or as a
    collection of it, and the prose that says which fields matter.
    `otherwise` are the kinds it produces instead under a condition the
    node states. An op and a protocol are declared in the same words
    (epic 000553): params, inputs, output(s)."""

    kind: str
    collection: bool = False
    doc: str = ""
    otherwise: tuple[Otherwise, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "collection": self.collection, "doc": self.doc}
        if self.otherwise:
            d["otherwise"] = [o.to_dict() for o in self.otherwise]
        return d


#: The old name of `Output`, read until 000565.
Emits = Output


class _Required:
    """Sentinel: the parameter has no default and must be given."""

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()


#: The words of the parameter type grammar. A param's `type` is a union
#: of alternatives separated by ` | `, each one of:
#:
#: * a word: `string`, `int`, `float`, `bool`, `null`;
#: * `selector` — a position selector, the `position` grammar;
#: * `model` — a model reference: a repository id, `{base, adapters}`,
#:   an endpoint `{provider, model}`, or a param of the protocol,
#:   `{"$param": "model"}`;
#: * `object` — a structure whose fields the param DECLARES (its own
#:   `fields`, or the `value` it names). An object nobody declared is
#:   what an editor can only show as JSON, so the suite refuses one;
#: * `json` — any JSON value, open on purpose: provider-native options,
#:   a metric's keyword arguments. The openness is in the type, where a
#:   reader sees it;
#: * `callable` — a Python function, for tests; never in a protocol;
#: * a quoted literal, `"all"`;
#: * `list[T]` — a list of `T`, itself a union;
#: * `map[string, T]` — string keys to `T`: `tracked`, `templates`.
TYPE_WORDS = frozenset({"string", "int", "float", "bool", "null",
                        "selector", "model", "object", "json", "callable",
                        # A reference to a stored object (epic 000553): the
                        # type of a protocol param whose value is an address
                        # — `{"$ref": {"bench": …}}` — rather than what is
                        # at it.
                        "ref"})


@dataclass(frozen=True)
class TypeNode:
    """One alternative of a parsed param type. `form` is `word`,
    `literal`, `list` or `map`; `word` is the word or the literal's
    text; `of` is a list's or a map's element union."""

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
    """A param's `type` as the union of its alternatives. Raises
    ValueError, naming the part, on anything outside the grammar."""
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
    """Every word a type uses, at any depth: `list[string | object]`
    uses `string` and `object`."""
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
    """One parameter of an op, or one field of a structured parameter.

    `type` is an expression in the grammar above, in the reader's terms
    — `int`, `list[string]`, `map[string, string]`, `"all" | list[int]`
    — not a Python annotation. `doc` is markdown; its first paragraph is
    the one-line description a table shows, and any further paragraphs
    are the details a page shows beneath it. `default` is the value the
    block uses when the param is absent, as the protocol would write it
    (JSON-shaped), or REQUIRED.
    """

    name: str
    type: str
    doc: str
    default: Any = REQUIRED
    #: The closed set a string takes, when it has one: `("layers",
    #: "positions")`. DECLARED and proved against the code's own check,
    #: never read out of the prose — a doc that shows `"resid_post"` and
    #: `"resid_pre"` may be naming two of a grammar's many points, and an
    #: editor that offered only those two would forbid the rest.
    choices: tuple[str, ...] = ()
    #: The shared `Value` this param's structure is, when it is one of
    #: the grammars many ops take: `"pool"`, `"point"`. Its fields (or
    #: its vocabulary) are declared once, there.
    value: str | None = None
    #: The fields of every `object` in this param's type, when the
    #: structure is this op's own: `lora` is `rank`, `alpha` and
    #: `target_modules`. Fields are params, so a field may be an object
    #: with fields of its own.
    fields: tuple[Param, ...] = ()
    #: The kind of stored object this param may be given BY REFERENCE
    #: (epic 000553): `{"$ref": {"bench": …}}` here is fetched by the
    #: executor at the node's boundary, recorded as a lineage input, and
    #: handed to the block as the value. A `$ref` anywhere this is not
    #: declared is refused before the node runs.
    stored: str | None = None
    #: The block wants the reference ITSELF — the address, unresolved — to
    #: stream from it lazily or to publish to it.
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

    A port is filled by an edge from an upstream node or from one of the
    protocol's inputs, or — for a small literal or a stored object —
    under the node's `inputs` map, as a list, an object, or `{"$ref":
    …}`. Never under `params`: what a node computes on is an input, what
    it computes with is a param.

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
    the node's `inputs`. `output`: the record it produces. `example`: a
    params object that would run, with `{"$param": …}` where the
    protocol's param supplies a value and `{"$ref": …}` where a stored
    object does; `example_inputs` the node's `inputs` beside it, when
    the example needs any.
    """

    #: The bare name, `family/op` — what a protocol writes.
    name: str
    summary: str
    description: str
    params: tuple[Param, ...]
    inputs: tuple[Port, ...] = ()
    #: The kind produced, or None for an op whose result is not a bench
    #: object (a tool handler's).
    output: Output | None = None
    #: What a machine must have to run this operation (task 000516).
    #: Declared here and nowhere else: the composer used to carry it per
    #: block in a hand-written table, and that table decides which
    #: runner may claim a job (`/jobs/next` filters on it), so a wrong
    #: entry routes work to a machine that cannot do it. It also drifted
    #: — `adapter/merge` was marked as needing local weights, and it
    #: never loads a model; its docstring says so.
    #:
    #: * `pure` — arithmetic over records and objects. No model, no
    #:   network; runnable anywhere, eventually API-side.
    #: * `mlx-local` — needs the weights resident on the machine.
    #: * `remote` — needs the network and the owner's credentials, but
    #:   no model: `adapter/publish` pushing to a hub.
    #: * `by-model` — whichever the `model` it is given needs: an
    #:   endpoint makes it remote, a repo makes it local. Chat and
    #:   judge are the same operation either way, which is the point of
    #:   them, and a single declared class would have to lie about one
    #:   of the two.
    requires: str = "pure"
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

    @property
    def emits(self) -> Output | None:
        """The old name of `output`, read until 000565."""
        return self.output

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
            # The old key, for a reader from before the rename (000565
            # removes it).
            "emits": self.output.to_dict() if self.output else None,
            "params": [p.to_dict() for p in self.params],
            "example": self.example,
            "example_inputs": self.example_inputs,
        }


def P(name: str, type: str, doc: str, default: Any = REQUIRED, *,
      choices: tuple[str, ...] = (), value: str | None = None,
      fields: tuple[Param, ...] = (), stored: str | None = None,
      reference: bool = False) -> Param:
    """Shorthand for a declaration file: `P("top_k", "int", "…", 5)`."""
    return Param(name, type, doc, default, choices, value, fields, stored, reference)


def In(name: str, kind: str, doc: str, *, required: bool = True,
       many: bool = False, variadic: bool = False,
       min_edges: int | None = None, max_edges: int | None = None,
       on_missing: str = "fail") -> Port:
    """Shorthand for a declaration file: `In("records", "records/record",
    "…", many=True)`."""
    return Port(name, kind, doc, required, many, variadic, min_edges,
                max_edges, on_missing)
