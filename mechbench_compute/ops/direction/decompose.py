from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.directions.constants import DEFAULT_AXIS
from mechbench_compute.directions.select_layer_items import select_layer_items
from mechbench_compute.directions.make import make
from mechbench_compute.directions.build_model_provenance import build_model_provenance
from mechbench_compute.directions.resolve_space import resolve_space
from mechbench_compute.lexicon._base import In, Op, Output, P

_VECTORS = In("vectors", "activations/vector",
              "A collection of vectors with items at the chosen `layer`.",
              many=True)


OP = Op(
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
    example={"layer": 14, "component": 0},
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/vectors"}}},
)


def run(ctx, inputs, params):
    vectors = inputs.get("vectors")
    value = params.get("value", params.get("label"))
    return fit_component(vectors, layer=int(params["layer"]),
                         component=int(params.get("component", 0)),
                         axis=str(params.get("axis") or DEFAULT_AXIS), value=value,
                         point=params.get("point"), source=params.get("source"))


def fit_component(vectors: Mapping[str, Any], *, layer: int, component: int = 0,
                  axis: str = DEFAULT_AXIS, value: Any = None, point: str | None = None,
                  source: str | None = None) -> dict[str, Any]:
    rows = select_layer_items(vectors, layer)
    if value is not None:
        rows = [r for r in rows if str(S.label_of(r, axis)) == str(value)]
    x = np.array([r["vector"] for r in rows], dtype=np.float32)
    if len(x) < 2:
        raise ValueError("PCA needs at least two items")
    x = x - x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    if component >= len(s):
        raise ValueError(f"component {component} out of range ({len(s)} available)")
    v = vt[component]
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    explained = float(s[component] ** 2 / max(float((s ** 2).sum()), 1e-12))
    return make(v, resolve_space(vectors, rows, point), method="pca",
                sources=[source] if source else [],
                labels=({"axis": axis, "value": value} if value is not None else None),
                extra={"component": int(component), "explained": round(explained, 4),
                       "n_items": len(x), **build_model_provenance(vectors, rows)})
