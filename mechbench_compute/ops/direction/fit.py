from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute.directions.constants import DEFAULT_AXIS
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
        P("point", "string",
          "Override the point recorded on the direction — `\"resid_post\"`, "
          "`\"resid_pre\"`, or any point name. By default it is taken from "
          "the vectors' own `space`.",
          None, value="point"),
        P("source", "string",
          "A label for where the vectors came from, recorded in the "
          "direction's derivation for provenance.",
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
                               point=params.get("point"), source=params.get("source"))


def fit_mean_difference(vectors: Mapping[str, Any], *, layer: int, positive: str,
                        negative: str, axis: str = DEFAULT_AXIS, point: str | None = None,
                        source: str | None = None) -> dict[str, Any]:
    """Difference of means: centroid(`positive`) − centroid(`negative`)
    at `layer`, the groups being the items' values on the `axis`
    coordinate."""
    rows = select_layer_items(vectors, layer)
    pos = np.array([r["vector"] for r in rows if str(S.label_of(r, axis)) == str(positive)],
                   dtype=np.float32)
    neg = np.array([r["vector"] for r in rows if str(S.label_of(r, axis)) == str(negative)],
                   dtype=np.float32)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            f"no items at layer {layer} with {axis}={positive!r}/{negative!r}")
    return make(pos.mean(0) - neg.mean(0), resolve_space(vectors, rows, point),
                method="diff_of_means", sources=[source] if source else [],
                labels={"axis": axis, "positive": positive, "negative": negative},
                extra={"n_positive": len(pos), "n_negative": len(neg),
                       **build_model_provenance(vectors, rows)})

