"""Ops that make, combine and read **directions**.

A direction is a unit vector in a model's activation space at one
(layer, point), carrying its own derivation: how it was made, from what,
on which model. One object flows everywhere a direction is used — into
`intervene/apply` (add, project out, clamp, rotate), `direction/project`,
`direction/unembed` — so a direction found one way can be tried every
other way without conversion.

The record is an `activations/vector` with a derivation: `{"kind":
"direction/vector", "space": {"model": "...", "layer": 14, "point":
"resid_post", "d": 2048}, "vector": [...], "norm": 37.2, "unit": true,
"derivation": {"method": "diff_of_means", "sources": [...], "model":
"...", ...}}`. `norm` is the magnitude before normalisation, which some
readings use.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import WILDCARD, Output, In, Op, P
from mechbench_compute.lexicon.model import ADAPTER

_DIRECTION = In("direction", "direction/vector", "The direction.")

_VECTORS = In("vectors", "activations/vector",
              "A collection of vectors with items at the chosen `layer`.",
              many=True)

_NAMED_DIRECTIONS = (
    In("directions", "direction/vector",
       "The directions as one list, when they do not each arrive on a port "
       "of their own; they are named `d0`, `d1`, … in the result.",
       many=True, required=False),
    In(WILDCARD, "direction/vector",
       "One direction per edge, on a port of your naming; the port name is "
       "the direction's name in the result.", required=False),
)


def _point() -> P:
    return P("point", "string",
             "Override the point recorded on the direction — `\"resid_post\"`, "
             "`\"resid_pre\"`, or any point name. By default it is taken from "
             "the vectors' own `space`.",
             None, value="point")


def _source() -> P:
    return P("source", "string",
             "A label for where the vectors came from, recorded in the "
             "direction's derivation for provenance.",
             None)


FROM_VECTORS = Op(
    name="direction/fit",
    summary=(
        "Make a direction from labelled residual vectors as the difference of "
        "two label centroids — the axis along which one group differs from "
        "the other."
    ),
    description="""\
Among the collection's items at `layer`, take the mean of those whose
`axis` coordinate is `positive` and subtract the mean of those at
`negative`. The result, normalised to unit length, points from the negative
group toward the positive one. The derivation records the axis, both values
and how many items went into each centroid; `norm` keeps the un-normalised
magnitude.

This is the difference-of-means method — the simplest and most robust way
to find a concept direction, and the one most steering results are built on.
""",
    inputs=(
        In("vectors", "activations/vector",
           "A collection of vectors with items at the chosen `layer`, grouped "
           "on the `axis` coordinate.", many=True),
    ),
    output=Output('direction/vector', collection=False, doc='`derivation.method` is `"diff_of_means"`, with `axis`, `positive`, `negative`, `n_positive` and `n_negative`.'),
    params=(
        P("layer", "int", "The layer whose items the centroids are taken from."),
        P("axis", "string",
          "The coordinate the items are grouped on. `label` reads the older "
          "`label` field as well.",
          "label"),
        P("positive", "string", "The `axis` value of the items the direction points toward."),
        P("negative", "string", "The `axis` value of the items the direction points away from."),
        _point(),
        _source(),
    ),
    example={
        "layer": 14,
        "axis": "register",
        "positive": "formal",
        "negative": "casual",
    },
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/vectors"}}},
)

FROM_REGRESSION = Op(
    name="direction/regress",
    summary=(
        "Make a direction by regressing residual vectors against a number the "
        "items carry — the axis along which a measured quantity rises."
    ),
    description="""\
`direction/fit` answers "which way does this group lie from that one": two
labels, a difference of centroids. Some signals are not two groups. A
token's surprisal, a passage's length, a judge's score — these are
quantities, and the question is which way the residual moves as the
quantity rises. That is a regression.

Among the items at `layer`, those carrying a number on the `target`
coordinate are fitted by ridge regression, choosing the penalty from
`alphas` by cross-validation. The fitted weight vector, normalised, is the
direction.

A weight vector always exists, so the fit holds out `holdout` of the items
and reports R² on them: `r2_test` is what says whether the direction
carries the signal or the fit merely memorised the training rows. The
derivation records the chosen `alpha`, both R²s, the correlation on the
held-out items, and the split. The common `seed` param fixes which items
are held out, so a fit repeats exactly.
""",
    inputs=(
        In("vectors", "activations/vector",
           "A collection of vectors with items at the chosen `layer`, each "
           "carrying the `target` number among its coordinates.", many=True),
    ),
    output=Output('direction/vector', collection=False,
                  doc='`derivation.method` is `"ridge"`, with `alpha`, `r2_train`, `r2_test`, `pearson_test`, `n_items`, `n_train`, `n_test` and `seed`.'),
    params=(
        P("layer", "int", "The layer whose items are fitted."),
        P("target", "string",
          "The coordinate holding the number to regress against. Items "
          "without one are left out, and the count of those kept is on the "
          "derivation."),
        P("alphas", "list[float]",
          "The ridge penalties to choose among, by cross-validation.",
          [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]),
        P("holdout", "float",
          "The fraction of items held out of the fit, to score it.", 0.2),
        _point(),
        _source(),
    ),
    example={"layer": 21, "target": "surprisal"},
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/residuals"}}},
)

FROM_PCA = Op(
    name="direction/decompose",
    summary=(
        "Make a direction from the principal component of a set of residual "
        "vectors — the axis along which they vary most."
    ),
    description="""\
The items at `layer` (optionally only those whose `axis` coordinate is
`value`) are centred and decomposed by SVD; the requested `component`
becomes the direction. A principal component has no intrinsic sign, so the
sign is fixed by making the largest-magnitude coordinate positive. The
derivation records the fraction of variance the component explains and the
number of items.

At least two items are needed.
""",
    inputs=(_VECTORS,),
    output=(
        Output('direction/vector', collection=False, doc='`derivation.method` is `"pca"`, with `derivation.component`, `derivation.explained` and `derivation.n_items`.')
    ),
    params=(
        P("layer", "int", "The layer whose items are decomposed."),
        P("component", "int",
          "Which principal component: `0` is the direction of greatest "
          "variance, `1` the next, and so on.",
          0),
        P("axis", "string",
          "The coordinate `value` is read on, when only one group is used.",
          "label"),
        P("value", "string",
          "Use only the items whose `axis` coordinate is this value. By "
          "default every item at the layer is used.",
          None),
        P("label", "string",
          "The older spelling of `value` on the `label` axis.",
          None),
        _point(),
        _source(),
    ),
    example={"layer": 14, "component": 0},
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/vectors"}}},
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
    inputs=_NAMED_DIRECTIONS,
    output=(
        Output('direction/vector', collection=False, doc='`derivation.method` is `"add"`, with `derivation.weights`.')
    ),
    params=(
        P("weights", "list[float]",
          "One coefficient per direction, in input order.",
          None),
    ),
    example={"weights": [1.0, 0.5]},
    example_inputs={"directions": [{"$ref": {"bench": "you/lab/formal"}}, {"$ref": {"bench": "you/lab/terse"}}]},
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
    inputs=_NAMED_DIRECTIONS,
    output=Output('direction/vector', collection=False, doc='`derivation.method` is `"average"`.'),
    params=(),
    example={},
    example_inputs={"directions": [{"$ref": {"bench": "you/lab/axis_a"}}, {"$ref": {"bench": "you/lab/axis_b"}}]},
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
        In("direction", "direction/vector", "The direction to clean."),
        In("against", "direction/vector",
           "The direction(s) to remove — one, or a list of them.", many=True),
    ),
    output=(
        Output('direction/vector', collection=False, doc='`derivation.method` is `"orthogonalize"`, with `derivation.against` (how many independent directions were removed).')
    ),
    params=(),
    example={},
    example_inputs={
        "direction": {"$ref": {"bench": "you/lab/sentiment"}},
        "against": [{"$ref": {"bench": "you/lab/length"}}],
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
    inputs=(In("direction", "direction/vector", "The direction to normalise."),),
    output=Output('direction/vector', collection=False, doc='`derivation.method` is `"normalize"`.'),
    params=(),
    example={},
    example_inputs={"direction": {"$ref": {"bench": "you/lab/imported"}}},
)

PROJECT = Op(
    name="direction/project",
    summary=(
        "Project every vector of a collection onto a direction — each "
        "prompt's scalar coordinate along that axis."
    ),
    description="""\
For each item of the collection at the direction's layer, the dot product
of the item's vector with the unit direction. Each coordinate keeps the
item's `id`, `coords` and `space`, so the result groups and plots the same
way the vectors did. A quick way to see whether a direction separates the
groups it was built from — or ones it was not.
""",
    inputs=(
        In("vectors", "activations/vector",
           "A collection of vectors with items at the direction's layer.",
           many=True),
        In("direction", "direction/vector", "The direction to project onto."),
    ),
    output=(
        Output('activations/coordinate', collection=True, doc="One item per input vector: `id`, `coords`, `space`, the `direction`'s identity and `coord`, the dot product with the unit direction.")
    ),
    params=(),
    example={},
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/vectors"}}, "direction": {"$ref": {"bench": "you/lab/axis"}}},
)

VOCAB = Op(
    name="direction/unembed",
    requires="mlx-local",
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
    inputs=(In("direction", "direction/vector", "The direction to read."), ADAPTER),
    output=(
        Output('direction/vocab', collection=False, doc="`space`, `top_k`, and `positive` and `negative` — each a distribution (`entropy_bits`, `top` as `{token, p, logp}`) of the unembedding applied to that sign.")
    ),
    params=(
        P("top_k", "int", "How many tokens to list per sign.", 10),
    ),
    example={"model": {"$param": "model"}, "top_k": 20},
    example_inputs={"direction": {"$ref": {"bench": "you/lab/axis"}}},
)

OPS: tuple[Op, ...] = (
    FROM_VECTORS, FROM_REGRESSION, FROM_PCA, ADD, AVERAGE, ORTHOGONALIZE, NORMALIZE, PROJECT,
    VOCAB,
)
