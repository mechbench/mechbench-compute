from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S
from mechbench_compute.directions.constants import DEFAULT_AXIS
from mechbench_compute.directions.items_at import _items_at
from mechbench_compute.directions.make import make
from mechbench_compute.directions.model_provenance import _model_provenance
from mechbench_compute.directions.space_at import _space_at
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.lexicon.direction import _point, _source

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
        _point(),
        _source(),
    ),
    example={"layer": 14, "component": 0},
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/vectors"}}},
)


def run(ctx, inputs, params):
    return block_from_pca(inputs, params)


def from_pca(vectors: Mapping[str, Any], *, layer: int, component: int = 0,
             axis: str = DEFAULT_AXIS, value: Any = None, point: str | None = None,
             source: str | None = None) -> dict[str, Any]:
    """A principal component of the (centered) items at `layer`,
    optionally only those whose `axis` coordinate is `value`. Sign is
    fixed so the largest-magnitude coordinate is positive (a component
    has no intrinsic sign)."""
    rows = _items_at(vectors, layer)
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
    return make(v, _space_at(vectors, rows, point), method="pca",
                sources=[source] if source else [],
                labels=({"axis": axis, "value": value} if value is not None else None),
                extra={"component": int(component), "explained": round(explained, 4),
                       "n_items": len(x), **_model_provenance(vectors, rows)})


def block_from_pca(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    vectors = inputs.get("vectors")
    # `label` is the retired spelling of `value` on the `label` axis.
    value = params.get("value", params.get("label"))
    return from_pca(vectors, layer=int(params["layer"]),
                    component=int(params.get("component", 0)),
                    axis=str(params.get("axis") or DEFAULT_AXIS), value=value,
                    point=params.get("point"), source=params.get("source"))
