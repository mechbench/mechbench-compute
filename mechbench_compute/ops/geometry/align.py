from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute.api import In, Op, Output, P, Resume, collection, items_of, read_header

METHODS = ("cka", "rsa")

OP = Op(
    name="geometry/align",
    resume=Resume("restart"),
    summary="CKA or RSA between two sets of similarity matrices over the same records, one score per pair of layers.",
    description="""\
Reads two `geometry/similarity` collections computed over the same records,
in the same order — two models, two prompts, two checkpoints — and reports
how alike their geometry is at every pair of layers. It compares matrices,
never vectors, so two models of different widths compare without sharing a
space.

`cka` is linear centred kernel alignment: each item's matrix is read as a
Gram matrix, double-centred when `center` is true, and the score is
`<K, L> / (|K| |L|)` under the Frobenius inner product, from 0 to 1. `rsa`
is the Spearman correlation between the two matrices' upper triangles,
from -1 to 1, ties ranked by their average.

A matrix whose header says its metric is a distance is read as the kernel
`-D²/2` for CKA and as `-D` for RSA, so larger means more alike on both
sides. Each input item must carry a `layer`; per-head groups are refused.
When both sets carry item `ids`, they must agree in value and order.
""",
    params=(
        P("method", "string",
          "Linear CKA on the Gram matrices, or Spearman RSA on the upper triangles.",
          "cka", choices=METHODS),
        P("center", "bool",
          "Double-centre each Gram matrix before CKA. RSA ranks the matrices as they are.",
          True),
    ),
    inputs=(
        In("a", "geometry/similarity", "The first set of similarity matrices, one item per layer."),
        In("b", "geometry/similarity", "The second set, over the same records in the same order."),
    ),
    output=Output("geometry/alignment", collection=True,
                  doc="One item per pair of layers `{id, a_layer, b_layer, score}`; the header carries "
                      "`method`, `center`, and the two input headers as `a` and `b`."),
    needs=frozenset(),
    example={"method": "cka", "center": True},
    example_inputs={"a": {"$ref": {"bench": "you/lab/similarity-gemma"}},
                    "b": {"$ref": {"bench": "you/lab/similarity-llama"}}},
)


def run(ctx, inputs, params):
    method = str(params.get("method") or "cka")
    if method not in METHODS:
        raise ValueError(f"geometry/align: method {method!r} is not one of {', '.join(METHODS)}")
    center = bool(params.get("center", True))
    rows = align_sets(inputs["a"], inputs["b"], method, center)
    return collection("geometry/alignment", rows, method=method, center=center,
                      a=read_input_header(inputs["a"]), b=read_input_header(inputs["b"]))


def read_input_header(source: Any) -> dict[str, Any]:
    return {k: v for k, v in read_header(source).items() if k not in ("kind", "item_kind", "key")}


def is_distance(source: Any) -> bool:
    return isinstance(source, Mapping) and source.get("metric_kind") == "distance"


def read_layers(source: Any, port: str) -> dict[int, tuple[list[Any] | None, np.ndarray]]:
    out: dict[int, tuple[list[Any] | None, np.ndarray]] = {}
    for item in items_of(source):
        layer = item.get("layer")
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise ValueError(f"geometry/align: an item on {port!r} ({item.get('group')!r}) carries no integer layer")
        if item.get("head") is not None or layer in out:
            raise ValueError(f"geometry/align: {port!r} has more than one matrix at layer {layer}; "
                             "compare one group per layer")
        matrix = np.asarray(item.get("matrix"), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError(f"geometry/align: the matrix at layer {layer} on {port!r} is not square with two rows")
        out[layer] = (item.get("ids"), matrix)
    if not out:
        raise ValueError(f"geometry/align: {port!r} carries no matrices")
    return out


def subtract_means(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True) - matrix.mean(axis=1, keepdims=True) + matrix.mean()


def score_cka(k: np.ndarray, l: np.ndarray, center: bool) -> float:
    if center:
        k, l = subtract_means(k), subtract_means(l)
    denominator = float(np.sqrt(np.sum(k * k) * np.sum(l * l)))
    return float(np.sum(k * l)) / denominator if denominator > 0 else 0.0


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def score_rsa(k: np.ndarray, l: np.ndarray) -> float:
    upper = np.triu_indices(k.shape[0], k=1)
    x, y = rank_values(k[upper]), rank_values(l[upper])
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y)) / denominator if denominator > 0 else 0.0


def read_kernel(matrix: np.ndarray, distance: bool, method: str) -> np.ndarray:
    if not distance:
        return matrix
    return -0.5 * matrix * matrix if method == "cka" else -matrix


def align_sets(a: Any, b: Any, method: str, center: bool) -> list[dict[str, Any]]:
    left, right = read_layers(a, "a"), read_layers(b, "b")
    rows: list[dict[str, Any]] = []
    for a_layer in sorted(left):
        a_ids, k = left[a_layer]
        k = read_kernel(k, is_distance(a), method)
        for b_layer in sorted(right):
            b_ids, l = right[b_layer]
            if k.shape != l.shape:
                raise ValueError(f"geometry/align: layer {a_layer} of a is {k.shape[0]} records and layer "
                                 f"{b_layer} of b is {l.shape[0]}; both must be over the same records")
            if a_ids is not None and b_ids is not None and list(a_ids) != list(b_ids):
                raise ValueError(f"geometry/align: layers {a_layer} of a and {b_layer} of b name different "
                                 "records, or the same records in another order")
            l = read_kernel(l, is_distance(b), method)
            score = score_cka(k, l, center) if method == "cka" else score_rsa(k, l)
            rows.append({"id": f"{a_layer}~{b_layer}", "a_layer": a_layer, "b_layer": b_layer,
                         "score": round(score, 6)})
    return rows
