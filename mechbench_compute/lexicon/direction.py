"""Ops that make, combine and read **directions**.

A direction is a unit vector in a model's activation space at one
(layer, point), carrying its own derivation: how it was made, from what,
on which model. One object flows everywhere a direction is used — into
`intervene/apply` (add, project out, clamp, rotate), `direction/project`,
`direction/vocab` — so a direction found one way can be tried every
other way without conversion.

The record: `{"kind": "direction/vector", "layer": 14, "point": "resid_post",
"d": 2048, "vector": [...], "norm": 37.2, "unit": true, "derivation":
{"method": "diff_of_means", "sources": [...], "model": "...", ...}}`.
`norm` is the magnitude before normalisation, which some readings use.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Emits, Op, P

_DIRECTION_IN = (
    "`direction` (by edge, or the `direction` param) — a direction record."
)

_DIRECTIONS_IN = """\
The directions come from every edge port carrying a direction record (the
port names become the names in the result), and/or from the `directions`
param as a list.
"""


def _point() -> P:
    return P("point", "string",
             "Override the point recorded on the direction. By default it is "
             "taken from the vectors record (`\"resid_post\"` for `point: "
             "\"post\"`).",
             None)


def _source() -> P:
    return P("source", "string",
             "A label for where the vectors came from, recorded in the "
             "direction's derivation for provenance.",
             None)


FROM_VECTORS = Op(
    name="direction/from-vectors",
    summary=(
        "Make a direction from labelled residual vectors as the difference of "
        "two label centroids — the axis along which one group differs from "
        "the other."
    ),
    description="""\
Among the vectors record's rows at `layer`, take the mean of those labelled
`positive` and subtract the mean of those labelled `negative`. The result,
normalised to unit length, points from the negative group toward the
positive one. The derivation records both labels and how many rows went
into each centroid; `norm` keeps the un-normalised magnitude.

This is the difference-of-means method — the simplest and most robust way
to find a concept direction, and the one most steering results are built on.
""",
    inputs=(
        "`vectors` (by edge, or the `vectors` param) — a `residual_vectors` "
        "record whose rows carry `label`, with rows at the chosen `layer`."
    ),
    emits=Emits('direction/vector', collection=False, doc='`derivation.method` is `"diff_of_means"`.'),
    params=(
        P("layer", "int", "The layer whose rows the centroids are taken from."),
        P("positive", "string", "The label of the rows the direction points toward."),
        P("negative", "string", "The label of the rows the direction points away from."),
        _point(),
        _source(),
    ),
    example={
        "vectors": {"$fetch": "$vectors"},
        "layer": 14,
        "positive": "formal",
        "negative": "casual",
    },
)

FROM_PCA = Op(
    name="direction/from-pca",
    summary=(
        "Make a direction from the principal component of a set of residual "
        "vectors — the axis along which they vary most."
    ),
    description="""\
The rows at `layer` (optionally only those with one `label`) are centred and
decomposed by SVD; the requested `component` becomes the direction. A
principal component has no intrinsic sign, so the sign is fixed by making
the largest-magnitude coordinate positive. The derivation records the
fraction of variance the component explains and the number of rows.

At least two rows are needed.
""",
    inputs=(
        "`vectors` (by edge, or the `vectors` param) — a `residual_vectors` "
        "record with rows at the chosen `layer`."
    ),
    emits=(
        Emits('direction/vector', collection=False, doc='`derivation.method` is `"pca"`, with `derivation.component`, `derivation.explained` and `derivation.n_rows`.')
    ),
    params=(
        P("layer", "int", "The layer whose rows are decomposed."),
        P("component", "int",
          "Which principal component: `0` is the direction of greatest "
          "variance, `1` the next, and so on.",
          0),
        P("label", "string",
          "Use only the rows with this label. By default every row at the "
          "layer is used.",
          None),
        _point(),
        _source(),
    ),
    example={"vectors": {"$fetch": "$vectors"}, "layer": 14, "component": 0},
)

ADD = Op(
    name="direction/add",
    summary=(
        "Combine several directions in the same space into one by weighted "
        "sum, re-normalised — steer along two concepts at once."
    ),
    description="""\
All inputs must share a layer, point and width; adding across spaces is
meaningless and is refused rather than producing a plausible-looking vector.
`weights` sets the mix (one per direction, default equal), so
`weights: [1, -0.5]` is "the first, minus half the second". The result is
normalised to unit length.
""",
    inputs=_DIRECTIONS_IN,
    emits=(
        Emits('direction/vector', collection=False, doc='`derivation.method` is `"add"`, with `derivation.weights`.')
    ),
    params=(
        P("directions", "list[direction]",
          "The directions to sum, when they do not arrive by edge.",
          None),
        P("weights", "list[float]",
          "One coefficient per direction, in input order.",
          None),
    ),
    example={
        "directions": [{"$fetch": "$formal"}, {"$fetch": "$terse"}],
        "weights": [1.0, 0.5],
    },
)

AVERAGE = Op(
    name="direction/average",
    summary=(
        "The mean of several unit directions in the same space — the "
        "component they share."
    ),
    description="""\
Each input is already unit length, so the mean weights every direction
equally however large its original norm was. That is the right question for
"what do these adapters' axes have in common?", where the answer must not be
dominated by whichever axis happened to be longest. Same as `direction/add`
with equal weights, except that the derivation says `average`.
""",
    inputs=_DIRECTIONS_IN,
    emits=Emits('direction/vector', collection=False, doc='`derivation.method` is `"average"`.'),
    params=(
        P("directions", "list[direction]",
          "The directions to average, when they do not arrive by edge.",
          None),
    ),
    example={"directions": [{"$fetch": "$axis_a"}, {"$fetch": "$axis_b"}]},
)

ORTHOGONALIZE = Op(
    name="direction/orthogonalize",
    summary=(
        "Remove from a direction its components along one or more other "
        "directions — what is left of a concept once a confound is taken "
        "out."
    ),
    description="""\
The `against` directions are orthonormalised (Gram–Schmidt) and the
direction's projection onto each is subtracted; the remainder is
re-normalised. A direction that lies entirely within the span of `against`
has nothing left and the block refuses it. All inputs must share a space.
""",
    inputs=(
        "`direction` (by edge or param) — the direction to clean. `against` "
        "(by edge or param) — one direction record or a list of them."
    ),
    emits=(
        Emits('direction/vector', collection=False, doc='`derivation.method` is `"orthogonalize"`, with `derivation.against` (how many independent directions were removed).')
    ),
    params=(
        P("direction", "direction",
          "The direction to clean, when it does not arrive by edge.",
          None),
        P("against", "direction | list[direction]",
          "The direction(s) to remove, when they do not arrive by edge.",
          None),
    ),
    example={
        "direction": {"$fetch": "$sentiment"},
        "against": [{"$fetch": "$length"}],
    },
)

NORMALIZE = Op(
    name="direction/normalize",
    summary=(
        "Rescale a direction to unit length, keeping its space and model — "
        "an explicit, recorded step for a vector that arrived some other way."
    ),
    description="""\
Directions made by the other `direction/*` ops are unit already. This is
for one that was hand-built or imported, and for making normalisation a
visible step in the graph rather than an assumption.
""",
    inputs=_DIRECTION_IN,
    emits=Emits('direction/vector', collection=False, doc='`derivation.method` is `"normalize"`.'),
    params=(
        P("direction", "direction",
          "The direction to normalise, when it does not arrive by edge.",
          None),
    ),
    example={"direction": {"$fetch": "$imported"}},
)

PROJECT = Op(
    name="direction/project",
    summary=(
        "Project every row of a residual-vectors record onto a direction — "
        "each prompt's scalar coordinate along that axis."
    ),
    description="""\
For each row of the vectors record at the direction's layer, the dot
product of the row's vector with the unit direction. The rows keep their
`id`, `label` and other fields, so the result groups and plots the same way
the vectors did. A quick way to see whether a direction separates the
labels it was built from — or ones it was not.
""",
    inputs=(
        "`vectors` (by edge, or the `vectors` param) — a `residual_vectors` "
        "record with rows at the direction's layer. `direction` (by edge or "
        "param) — the direction."
    ),
    emits=(
        Emits('activations/coordinate', collection=True, doc='One item per input vector: the item without its `vector`, plus `projection`. The header carries `layer` and `point`.')
    ),
    params=(
        P("direction", "direction",
          "The direction to project onto, when it does not arrive by edge.",
          None),
    ),
    example={"vectors": {"$fetch": "$vectors"}, "direction": {"$fetch": "$axis"}},
)

SIMILARITY = Op(
    name="direction/similarity",
    summary=(
        "The cosine between two directions, or the full pairwise cosine "
        "matrix over many — are these axes the same axis?"
    ),
    description="""\
Given exactly `a` and `b`, one cosine. Given any other set of directions
(by edge ports and/or the `directions` list), the pairwise matrix — the
question "are these eight adapters' axes aligned?" in one node instead of
twenty-eight. Directions are named by their port (or `d0`, `d1`, … from the
list), and each one's original `norm` rides along. All must share a space.
""",
    inputs=(
        "Either `a` and `b` (by edge or param), or any set of direction "
        "ports and/or the `directions` param."
    ),
    emits=Emits('geometry/similarity', collection=False, doc='For two directions: `cosine`. For many: `names`, `cosines` (the matrix), `norms` and `pairs` (every pair with its cosine, most similar first). `metric` is `cosine` either way.'),
    params=(
        P("a", "direction", "The first of exactly two directions.", None),
        P("b", "direction", "The second of exactly two directions.", None),
        P("directions", "list[direction]",
          "Several directions for the pairwise matrix, when they do not "
          "arrive by edge.",
          None),
    ),
    example={"a": {"$fetch": "$axis_run1"}, "b": {"$fetch": "$axis_run2"}},
)

VOCAB = Op(
    name="direction/vocab",
    summary=(
        "Read a direction through the model's unembedding: the tokens it "
        "promotes and the tokens its negative promotes — what the axis "
        "'says' in vocabulary."
    ),
    description="""\
The direction (and its negative) is passed through the model's final norm
and unembedding as if it were a residual state, and the most probable tokens
in each sign are listed. The final norm is scale-invariant, so a unit
direction reads the same as any multiple of it.

The reading is only literal for directions at the residual stream; a
direction inside an attention block is not in the space the unembedding
reads.
""",
    inputs=_DIRECTION_IN,
    emits=(
        Emits('direction/vocab', collection=False, doc='`positive` and `negative`, each a list of `{token, p}` for the top tokens of that sign.')
    ),
    params=(
        P("direction", "direction",
          "The direction to read, when it does not arrive by edge.",
          None),
        P("top_k", "int", "How many tokens to list per sign.", 10),
    ),
    example={"model": "$model", "direction": {"$fetch": "$axis"}, "top_k": 20},
)

OPS: tuple[Op, ...] = (
    FROM_VECTORS, FROM_PCA, ADD, AVERAGE, ORTHOGONALIZE, NORMALIZE, PROJECT,
    SIMILARITY, VOCAB,
)
