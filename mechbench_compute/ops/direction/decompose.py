from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.api import collection
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
        "Make directions from the principal components of a set of residual "
        "vectors — the axes along which they vary most."
    ),
    description="""\
The items at `layer` (optionally only those whose `axis` coordinate is
`value`) are centred and decomposed by SVD. A principal component has no
intrinsic sign, so each one's sign is fixed by making its largest-magnitude
coordinate positive. Each direction's derivation records the fraction of
the variance it explains and the number of items.

There are two forms. `component` makes one direction, the component it
names (the first by default). `k` makes a collection of the first `k`,
greatest variance first, each carrying its `component` index as a
coordinate; the header carries what no single direction can: how much of
the variance the `k` explain together, at which layer and point, out of
how many vectors. `activations/examples` reads one direction, so to see
what turns each component on, map over the collection with `records/map`.

`k` and `component` together are refused: the one form or the other.

At least two items are needed.
""",
    inputs=(_VECTORS,),
    output=(
        Output('direction/vector', collection=False, doc='`derivation.method` is `"pca"`, with `derivation.component`, `derivation.explained` and `derivation.n_items`. With `k`, a collection of `k` of them, ids `pc0`, `pc1`, …, ordered by `coords.component`; the header carries `components` (`k`), `explained` (the sum of their shares), `layer`, `point`, `model` and `n_items`.')
    ),
    params=(
        P("layer", "int", "The layer whose items are decomposed."),
        P("component", "int",
          "Which principal component, for one direction: `0` is the "
          "direction of greatest variance, `1` the next, and so on. `0` "
          "when neither this nor `k` is given.",
          None),
        P("k", "int",
          "How many principal components to make, greatest variance first, "
          "as a collection. In place of `component`.",
          None),
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
    example={"layer": 14, "k": 8},
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/vectors"}}},
)


def run(ctx, inputs, params):
    vectors = inputs.get("vectors")
    value = params.get("value", params.get("label"))
    k, component = params.get("k"), params.get("component")
    common = dict(layer=int(params["layer"]), axis=str(params.get("axis") or DEFAULT_AXIS),
                  value=value, point=params.get("point"), source=params.get("source"))
    if k is not None:
        if component is not None:
            raise ValueError(
                "direction/decompose takes `k` (a collection of the first k "
                "components) or `component` (one direction), not both")
        return fit_components(vectors, k=int(k), **common)
    return fit_component(vectors, component=int(component or 0), **common)


def _decompose(vectors: Mapping[str, Any], layer: int, axis: str,
               value: Any) -> tuple[list[Mapping[str, Any]], int, np.ndarray, np.ndarray]:
    rows = select_layer_items(vectors, layer)
    if value is not None:
        rows = [r for r in rows if str(S.label_of(r, axis)) == str(value)]
    x = np.array([r["vector"] for r in rows], dtype=np.float32)
    if len(x) < 2:
        raise ValueError("PCA needs at least two items")
    x = x - x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    return rows, len(x), s, vt


def _choose_sign(v: np.ndarray) -> np.ndarray:
    return -v if v[np.argmax(np.abs(v))] < 0 else v


def _measure_share(s: np.ndarray, i: int) -> float:
    return round(float(s[i] ** 2 / max(float((s ** 2).sum()), 1e-12)), 4)


def fit_component(vectors: Mapping[str, Any], *, layer: int, component: int = 0,
                  axis: str = DEFAULT_AXIS, value: Any = None, point: str | None = None,
                  source: str | None = None) -> dict[str, Any]:
    rows, n, s, vt = _decompose(vectors, layer, axis, value)
    if component >= len(s):
        raise ValueError(f"component {component} out of range ({len(s)} available)")
    return make(_choose_sign(vt[component]), resolve_space(vectors, rows, point), method="pca",
                sources=[source] if source else [],
                labels=({"axis": axis, "value": value} if value is not None else None),
                extra={"component": int(component), "explained": _measure_share(s, component),
                       "n_items": n, **build_model_provenance(vectors, rows)})


def fit_components(vectors: Mapping[str, Any], *, layer: int, k: int,
                   axis: str = DEFAULT_AXIS, value: Any = None, point: str | None = None,
                   source: str | None = None) -> dict[str, Any]:
    rows, n, s, vt = _decompose(vectors, layer, axis, value)
    if not 1 <= k <= len(s):
        raise ValueError(f"k is between 1 and {len(s)} here (the items and their width "
                         f"bound it), not {k}")
    sp = resolve_space(vectors, rows, point)
    provenance = build_model_provenance(vectors, rows)
    items = []
    for i in range(k):
        item = make(_choose_sign(vt[i]), sp, method="pca",
                    sources=[source] if source else [],
                    labels=({"axis": axis, "value": value} if value is not None else None),
                    extra={"component": i, "explained": _measure_share(s, i),
                           "n_items": n, **provenance})
        items.append({"id": f"pc{i}", **item, "coords": {"component": i}})
    explained = round(sum(it["derivation"]["explained"] for it in items), 4)
    return collection(
        "direction/vector", items, order_by=["coords.component"],
        components=k, explained=explained, layer=sp.get("layer"), point=sp.get("point"),
        model=sp.get("model"), n_items=n,
        description=(f"The first {k} principal components of {n} vectors at layer "
                     f"{sp.get('layer')} {sp.get('point')}: {explained:.1%} of their variance."))
