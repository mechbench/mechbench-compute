from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.directions.constants import DEFAULT_AXIS
from mechbench_compute.directions.drop_dimensions import drop_dimensions
from mechbench_compute.directions.read_exclude import read_exclude
from mechbench_compute.directions.select_layer_items import select_layer_items
from mechbench_compute.directions.make import make
from mechbench_compute.directions.build_model_provenance import build_model_provenance
from mechbench_compute.directions.resolve_space import resolve_space
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
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

`layer` has to be chosen before the run, and the best layer is usually not
known yet. The usual pattern is two runs of one protocol. It captures
several layers and feeds them to both `direction/classify` and this op,
with `layer` as a protocol param. The first run's probe reports accuracy
by layer, and the second run binds `layer` to the best of them. [Protocols
as files](/protocol-files/) has a complete protocol built this way.

Some models keep a massive-activation dimension: one coordinate of the
residual runs hundreds of times larger than the rest, at nearly every
position and layer after the first few (Gemma 3's is 443). The trace of
that dimension across layers is the usual tell, and so is a dictionary
whose `b_dec` is almost all one coordinate. A small relative change in
it can be most of the difference between the centroids, so the axis,
its `norm`, and every cosine taken with it then measure that one
coordinate. `exclude` leaves named dimensions out: they are zero in the
vector before it is normalised, so a cosine or norm downstream agrees
with one computed by hand over the other dimensions, and
`excluded_share` records how much of the squared difference they
carried.

`center` removes each vector's own mean over its dimensions (the ones
not excluded) before the centroids are taken. That takes out an offset
shared by every coordinate of a vector, which a difference of means
keeps when the two groups carry different offsets. Centring each
dimension over the items would not change a difference of means at all,
so it is not offered. Use `center` when the reading that follows treats
the residual the way a centring layer norm does; leave it off when the
model's norm only rescales, as RMSNorm does. Centring does not take a
massive dimension out of the axis; `exclude` does.
""",
    inputs=(
        In("vectors", "activations/vector",
           "A collection of vectors with items at the chosen `layer`, grouped "
           "on the `axis` coordinate.", many=True),
    ),
    output=Output('direction/vector', collection=False,
                  doc='`derivation.method` is `"diff_of_means"`, with `axis`, `positive`, `negative`, '
                      '`n_positive` and `n_negative`; `center: true` when the vectors were centred, and '
                      '`exclude` and `excluded_share` when dimensions were left out.'),
    params=(
        P("layer", "int", "The layer whose items the centroids are taken from."),
        P("axis", "string",
          "The coordinate the items are grouped on. `label` reads the older "
          "`label` field as well.",
          "label"),
        P("positive", "string", "The `axis` value of the items the direction points toward."),
        P("negative", "string", "The `axis` value of the items the direction points away from."),
        P("point", "string",
          "Override the point recorded on the direction — `\"resid_post\"`, "
          "`\"resid_pre\"`, or any point name. By default it is taken from "
          "the vectors' own `space`.",
          None, value="point"),
        P("source", "string",
          "A label for where the vectors came from, recorded in the "
          "direction's derivation for provenance.",
          None),
        P("center", "bool",
          "Remove each vector's mean over its dimensions before the centroids are taken.",
          False),
        P("exclude", "list[int]",
          "Dimensions left out: zero in the direction, and recorded on its derivation.",
          None),
    ),
    example={
        "layer": 14,
        "axis": "register",
        "positive": "formal",
        "negative": "casual",
    },
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/vectors"}}},
)


def run(ctx, inputs, params):
    vectors = inputs.get("vectors")
    return fit_mean_difference(vectors, layer=int(params["layer"]),
                               axis=str(params.get("axis") or DEFAULT_AXIS),
                               positive=str(params["positive"]), negative=str(params["negative"]),
                               point=params.get("point"), source=params.get("source"),
                               center=bool(params.get("center") or False),
                               exclude=params.get("exclude"))


def fit_mean_difference(vectors: Mapping[str, Any], *, layer: int, positive: str,
                        negative: str, axis: str = DEFAULT_AXIS, point: str | None = None,
                        source: str | None = None, center: bool = False,
                        exclude: list[int] | None = None) -> dict[str, Any]:
    rows = select_layer_items(vectors, layer)
    pos = np.array([r["vector"] for r in rows if str(S.label_of(r, axis)) == str(positive)],
                   dtype=np.float32)
    neg = np.array([r["vector"] for r in rows if str(S.label_of(r, axis)) == str(negative)],
                   dtype=np.float32)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            f"no items at layer {layer} with {axis}={positive!r}/{negative!r}")
    dropped = read_exclude(exclude, pos.shape[1])
    shaped: dict[str, Any] = {}
    if center:
        shaped["center"] = True
    if dropped:
        raw = pos.mean(0) - neg.mean(0)
        total = float(np.dot(raw.astype(np.float64), raw.astype(np.float64)))
        part = float(np.sum(raw[dropped].astype(np.float64) ** 2))
        shaped["exclude"] = dropped
        shaped["excluded_share"] = round(part / total, 6) if total else 0.0
    if center or dropped:
        pos, neg = drop_dimensions(pos, dropped, center=center), drop_dimensions(neg, dropped, center=center)
    return make(pos.mean(0) - neg.mean(0), resolve_space(vectors, rows, point),
                method="diff_of_means", sources=[source] if source else [],
                labels={"axis": axis, "positive": positive, "negative": negative},
                extra={"n_positive": len(pos), "n_negative": len(neg), **shaped,
                       **build_model_provenance(vectors, rows)})
