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

from dataclasses import dataclass
from typing import Any


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


@dataclass(frozen=True)
class Op:
    """One canonical operation, described for the person using it.

    `summary`: one sentence — what the op does, in plain words. It is
    what an index and `llms.txt` show, so it has to stand alone.
    `description`: markdown, as long as it needs to be — what the op
    does, the shapes it expects, what the result looks like.
    `inputs`: what arrives by edge or by the common wiring params.
    `emits`: the record it produces. `example`: a params object that
    would run, with `$bindings` and `{"$fetch": …}` where a value comes
    from outside the protocol.
    """

    ref: str
    summary: str
    description: str
    params: tuple[Param, ...]
    inputs: str = ""
    emits: str = ""
    example: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        """`~canonical/ops/trajectory/capture/1` -> `trajectory/capture`."""
        return self.ref.removeprefix("~canonical/ops/").rsplit("/", 1)[0]

    @property
    def param_names(self) -> frozenset[str]:
        return frozenset(p.name for p in self.params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "name": self.name,
            "version": self.ref.rsplit("/", 1)[-1],
            "summary": self.summary,
            "description": self.description,
            "inputs": self.inputs,
            "emits": self.emits,
            "params": [p.to_dict() for p in self.params],
            "example": self.example,
        }


def P(name: str, type: str, doc: str, default: Any = REQUIRED) -> Param:
    """Shorthand for a declaration file: `P("top_k", "int", "…", 5)`."""
    return Param(name, type, doc, default)
