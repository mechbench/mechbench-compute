from __future__ import annotations

from typing import Any

from mechbench_compute import points as _points
from mechbench_compute.lexicon._base import Value


def F(type_: str, doc: str, **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"type": type_, "description": doc}
    d.update(extra)
    return d


ID = F("string", "The record's identity within its collection.")
COORDS = F("object", "The experimental coordinates the record belongs to: axis name → level key.",
           additionalProperties={"type": ["string", "integer", "number"]})
SPACE_DOC = ("The activation space the vector lives in: `{model, layer | null, point, head | null, d}`; "
             "`point` is one of the forward pass's point names (`resid_post`, `attn_out`, `attn.q`, …). "
             "Two vectors are comparable only when their spaces agree.")
TOKEN = F("object", "A token as `{id, text}`.", properties={"id": {"type": "integer"}, "text": {"type": "string"}})
VEC = F("array", "A dense float vector; position is the only key.", items={"type": "number"})
SPACE = F("object", SPACE_DOC,
          properties={"model": {"type": ["string", "null"]}, "layer": {"type": ["integer", "null"]},
                      "point": {"type": "string"}, "head": {"type": ["integer", "null"]}, "d": {"type": "integer"}},
          required=["model", "layer", "point", "head", "d"])
TOP = F("array", "The most likely tokens, ranked by probability, each `{token, p, logp}`.",
        items={"type": "object", "properties": {"token": TOKEN, "p": {"type": "number"}, "logp": {"type": "number"}}})
TRACKED = F("object", "Name → `{token, p, logp, variants}` for the answers the caller asked about, by the names it gave.",
            additionalProperties={"type": "object"})
VARIANTS = F("array", "Each spelling of a tracked answer, with and without a leading space, as `{token, p, logp}`; one entry when both are the same token.",
             items={"type": "object", "properties": {"token": TOKEN, "p": {"type": "number"}, "logp": {"type": "number"}}})


SPACE_VALUE = Value(
    "space",
    "The activation space a vector lives in: which model, which layer and point of the forward pass, which head, and how wide.",
    fields={
        "model": F("string | null", "The model the vector was read from, in its reference form; null for a direction fit across models."),
        "layer": F("integer | null", "The layer, counting from 0; null at a whole-model point (`embed`, `final_norm`, `logits`)."),
        "point": F("string", "Where in the forward pass: one name from the point vocabulary."),
        "head": F("integer | null", "The attention head, for a per-head point; null otherwise."),
        "d": F("integer", "The vector's width."),
    },
    required=("model", "layer", "point", "head", "d"),
    doc="""\
Every `activations/vector` — and so every direction and every trajectory
point — carries its space, all five fields always present. A space is the
thing two vectors must share to be compared: every vector metric refuses
two whose spaces differ, naming both, because a cosine between a layer-12
residual and a layer-13 residual is a number that means nothing.

A direction fit from vectors of more than one model — a base capture and
an adapted capture in one union, whose difference of means is the axis
along which the adapter moved the model — lives in neither model. Its
`model` is null, the residual basis the two share, and its `derivation`
names both models. A null model agrees with any, so such axes compare with
each other and with a direction from either model.

Collections of vectors are grouped by space when compared: `geometry/
similarity` with `by: "space"` builds one matrix per layer (and per head,
when heads are present), so a capture at several layers yields several
matrices rather than one that mixes them.
""",
)

TOKEN_VALUE = Value(
    "token",
    "One token of a model's vocabulary, as its id and its decoded text.",
    fields={
        "id": F("integer", "The token's id in the model's vocabulary."),
        "text": F("string", "The decoded text, including any leading space."),
    },
    required=("id", "text"),
    doc="""\
A token is always carried as both: the id is what the model computed over,
the text is what a person reads. The text includes the leading space when
the tokenizer's piece has one — `" Paris"` and `"Paris"` are two different
tokens, and the difference is usually the whole finding.

Wherever a protocol names an answer to follow — the `tracked` parameter,
a record's `target` or `outcomes` — it is looked for both with a leading
space and without one, whichever way it was written: `" Paris"` and
`"Paris"` name the same answer, the pair of tokens `" Paris"` and `"Paris"`.
Which of the two a model says first depends on what precedes it (after a
chat template's assistant prefix it is usually the bare one), so the answer
counts either way. An answer that tokenizes to several pieces is followed
by its first piece, and the read says so.
""",
)

TOP_VALUE = Value(
    "top",
    "The most likely next tokens, ranked by probability, each with its probability and log-probability.",
    fields={
        "token": F("object", "The token, as `{id, text}`."),
        "p": F("number", "Its probability under the distribution."),
        "logp": F("number", "Its natural log-probability."),
    },
    required=("token", "p", "logp"),
    doc="""\
A ranked list: the most probable token first. How many it holds is the
`top_k` on the collection's header. Two tokens with the same probability
are ordered by token id, so the list is the same on every run — a tie at
the top is a fact about the model, and the read records it as one rather
than choosing by chance.

`top` is a summary of the distribution, not the distribution: the mass
outside it is whatever `1 − Σ p` leaves. A metric between two
distributions compares over the union of the tokens both carry and treats
the unnamed remainder as one last bucket.
""",
)

TRACKED_VALUE = Value(
    "tracked",
    "The tokens a read was asked to follow, by the names the protocol gave them, each with its probability.",
    fields={
        "<name>": F("object", "`{token, p, logp, variants}` for the answer the name resolved to."),
    },
    doc="""\
A map from the caller's names to what the model said about them. The names
come from the op's `tracked` parameter (`{"answer": " Paris"}`), or from
the record's own `tracked` — which takes precedence, so a design can name
different outcomes per condition — or from a record's `outcomes` when it
carries those instead.

The first entry is the **target**: the token a sweep's change in
log-probability is taken on, the token a lens follows through the layers.
With nothing named, the target is the model's own top-1 prediction for
that prompt, and the read names it so.

An answer is the set of its spellings with and without a leading space.
`p` is their summed probability — the chance the model says the answer
either way — and `logp` its log; `token` is the spelling this read gives
more probability, and `variants` lists each spelling's own `{token, p,
logp}`, one entry when both spellings are the same token. A change in
log-probability is always taken on the set, on both sides; a rank is the
better spelling's; a logit, which belongs to one token, is the preferred
spelling's, and the op names it.

A name whose token is nowhere in the model's ranking still appears, with
whatever probability the model gave it; a read never drops an outcome
because it was unlikely.
""",
)

COORDS_VALUE = Value(
    "coords",
    "The experimental coordinates a record belongs to: which level of which factor it is.",
    fields={
        "<axis>": F("string | integer | number", "The record's level on that axis, by the level's key."),
    },
    doc="""\
`coords` is how a record knows where in the design it sits. `records/
cross` writes one record per combination of factors, each with `coords`
`{"genre": "noir", "seed": 1}`; every op that reads records copies the
coordinates onto what it produces, so a decision read, a captured vector and
a verdict all still know their genre. A grouping is always a coordinate:
the ops that group or separate take an `axis` naming one, never a field
of their own.

Two coordinates are written by ops rather than by the design. `records/
union` stamps each record with the name of the port it came from, on the
`batch` axis (or the one its `batch_axis` names), so a union of a base
capture and an adapted capture can be split again by that axis. A
grouping op's own output carries the group's value on the axis it
grouped by.
""",
)

DERIVATION_VALUE = Value(
    "derivation",
    "How a direction was made: the method, the vectors and labels it was fit on, and the models they came from.",
    fields={
        "method": F("string", "`diff_of_means`, `pca`, `add`, `average`, `orthogonalize` or `normalize`."),
        "sources": F("array", "The labels the protocol gave the inputs, when it gave any.", items={"type": "string"}),
        "model": F("string | null", "The model of the direction's space; null when fit across models."),
        "models": F("array", "Every model the fitting vectors came from, when more than one.", items={"type": "string"}),
        "axis": F("string", "For a difference of means: the coordinate the groups were read on."),
        "positive": F("string | number", "For a difference of means: the level whose centroid is the plus side."),
        "negative": F("string | number", "For a difference of means: the level whose centroid is the minus side."),
        "n_positive": F("integer", "How many vectors the plus centroid averaged."),
        "n_negative": F("integer", "How many vectors the minus centroid averaged."),
        "component": F("integer", "For a principal component: which one, counting from 0."),
        "explained": F("number", "For a principal component: the share of variance it carries."),
    },
    required=("method",),
    doc="""\
A direction is a unit vector plus the record of how it came to be, and the
record is what makes one direction comparable to another in a reader's
mind: two axes with the same `method`, `axis` and levels are the same
measurement on two models; two with different methods are not, however
close their cosine.

The arithmetic ops carry the derivation forward — an `average` names the
method and keeps the inputs' sources — so a direction that has been
orthogonalised against a confound still says what it was before.
""",
)

POSITION_VALUE = Value(
    "position",
    "The one selector that names token positions wherever an operation chooses where to read or edit.",
    grammar=True,
    doc="""\
A selector names positions of a rendered sequence:

| Selector | Positions |
|---|---|
| `"last"` | the final position — a decision point, after any prefill |
| `"all"` | every position |
| `[3, 5, -1]` | those indices; a negative index counts from the end |
| `{"tokens": ["x", "y"]}` | every position whose token text is one of the names |
| `{"range": [a, b]}` | positions `a` … `b − 1`; a null or negative end reads as a slice |
| `{"after": n}` | positions `n` … end |
| `{"segment": "thinking"}` | a named span of the trace — see below |
| `"subject"` | the last token of the record's `subject` string |
| `"generated"` | from where generation began — the trace's span, or the end of the rendered prompt |

A parameter that reads one position (`position` on `activations/capture`,
on `intervene/steer`, on a capture readout) takes the same selector and
must resolve to exactly one; a selector that names several there is
refused with the count. A selector that names nothing — a token text the
prompt lacks, a range past its end — is refused rather than read as
empty.

Positions count over the rendered sequence: the chat template's own
tokens are positions too, which is why `"last"` and `"generated"` are
usually the right words and a bare index rarely is.

`{"segment": role}` names a span the document itself declares. Generation
writes `prompt` and `body` always, and `thinking` and `answer` when the
model's own vocabulary declares reasoning delimiters — `<think>` and
`</think>` are tokens in Qwen3's and the R1 distills' vocabularies, not
prose they happen to write. So a capture may read the
residual stream *while the model reasoned*, and a steer may act there
and nowhere else, without any operation learning a new word.

A document that has no such span refuses the selector and says which
roles it does have. That is the point: a model which declares no
delimiters, or wrote none this time, has no reasoning to read, and a
capture aimed at it must not quietly return an answer instead.
""",
)

POOL_VALUE = Value(
    "pool",
    "The one pooling clause: read a set of positions and reduce them to one vector.",
    grammar=True,
    fields={
        "reduce": F("string", "`mean` or `max`, applied element-wise over the selected positions.",
                    choices=["mean", "max"], default="mean"),
        "over": F("selector", "Which positions to pool: a position selector; `\"all\"` by default.",
                  default="all"),
    },
    doc="""\
`pool: {"reduce": "mean", "over": "all"}` turns a one-position read into
a reduction over the positions `over` selects, so a whole text can be
embedded as one vector rather than represented by its last token. The
vector records `n_pooled`, how many positions went into it; a pooled
vector that does not say its own n cannot be audited.

On a trajectory the same clause pools over **steps**: `over` selects
steps along the trajectory's axis, and the point records the window as
`steps`. The spellings from before the clause was one — a first-k after
a skip, a last-k — are `{"range": [skip, skip + k]}` and `{"range":
[-k, null]}`.
""",
)

_POINT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("The residual stream", ("resid_pre", "resid_post")),
    ("Sub-layer outputs, added into the residual", ("attn_out", "mlp_out", "gate_out")),
    ("Inside the attention block", (
        "attn.in_norm", "attn.q_pre_norm", "attn.k_pre_norm", "attn.q_pre_rope", "attn.k_pre_rope",
        "attn.q", "attn.k", "attn.v", "attn.scores", "attn.weights", "attn.per_head_out", "attn.o_in")),
    ("Inside the MLP block", ("mlp.in_norm", "mlp.gate", "mlp.up", "mlp.act", "mlp.down_in")),
    ("Once per forward pass", ("embed", "final_norm", "logits")),
)


def _point_table() -> str:
    lines = ["| Where | Points |", "|---|---|"]
    for title, names in _POINT_GROUPS:
        lines.append(f"| {title} | " + ", ".join(f"`{n}`" for n in names) + " |")
    return "\n".join(lines)


POINT_VALUE = Value(
    "point",
    "The one vocabulary for where in a forward pass an activation is read or edited.",
    grammar=True,
    choices=tuple(n for _, names in _POINT_GROUPS for n in names),
    doc=f"""\
A **point** is a place in the forward pass. Every `point` parameter, every
`space.point`, and the hook names a capture readout reports use these
names and no others; an unknown name is refused with the list.

{_point_table()}

`resid_pre` is the residual stream entering a layer and `resid_post` the
stream leaving it; the sub-layer outputs are what each block adds. The
points inside a block are read per head where the tensor has heads
(`attn.q`, `attn.k`, `attn.v`, `attn.scores`, `attn.weights`,
`attn.per_head_out`) and a `space` read there carries the head. The three
whole-model points occur once per pass and take no `layers`.

An op that reads a residual stream (`activations/capture`, `trajectory/
capture`, `activations/contrast`) takes only the two residual points.
`intervene/ablate-layers` zeroes sub-layer outputs — `attn_out`, `mlp_out`,
`gate_out`, both of the first two by default, which removes the whole
layer's contribution. `intervene/apply` edits at any point.
""",
)

VALUES: tuple[Value, ...] = (
    SPACE_VALUE, TOKEN_VALUE, TOP_VALUE, TRACKED_VALUE, COORDS_VALUE, DERIVATION_VALUE,
    POSITION_VALUE, POOL_VALUE, POINT_VALUE,
)

BY_VALUE: dict[str, Value] = {v.name: v for v in VALUES}

DOCUMENTED_POINTS: frozenset[str] = frozenset(n for _, names in _POINT_GROUPS for n in names)
assert DOCUMENTED_POINTS == frozenset(_points.POINTS), (
    sorted(DOCUMENTED_POINTS ^ frozenset(_points.POINTS)))

__all__ = [
    "BY_VALUE", "COORDS", "DOCUMENTED_POINTS", "F", "ID", "SPACE", "SPACE_DOC", "TOKEN", "TOP",
    "TRACKED", "VALUES", "VEC",
]
