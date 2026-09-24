from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from mechbench_compute import metrics as M
from mechbench_compute import shapes as S
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="geometry/compare",
    summary=(
        "The pairwise matrix of a collection's items under a metric their "
        "kind declares — cosine over vectors, Jensen–Shannon over decision "
        "reads, hamming over records — with how well the groups separate."
    ),
    description="""\
A kind declares how its items compare, the way it declares how they are
drawn: `activations/vector` (and so directions and trajectory points) by
`cosine` (option `center`), `euclidean` or `dot`; `logits/distribution`
(and so decision reads, funnels and readouts) by `jensen-shannon`,
`hellinger`, `total-variation` or `kl`; `records/record` (and every record
kind) by `hamming` over `coords`. The op takes any such collection on
`items`, applies the named `metric` — the kind's first when none is named —
and produces the matrix with the metric, its `options`, and whether it is
symmetric recorded on the header.

Items are grouped `by` a header axis before comparing: `"space"` (the
default for vectors) compares only items from one layer and head; a
coordinate name (`"layer"` for a funnel, `"factor"` for a readout) compares
within each of its values; `null` compares everything at once. One item per
group, with every pair listed for groups of at most thirty-two.

When every item in a group has a value on the `axis` coordinate and there is
more than one value, the item also reports the mean value within groups,
between groups, their gap, the fraction of items whose nearest neighbour
shares their value, and the silhouette score.

Two vectors compare only within one space; two decision reads compare over
the union of the tokens they carry, the mass neither names counted as one
last bucket. Raw cosine between transformer activations is dominated by a
shared direction they all lean toward; for a variety measure use `cosine`
with `options: {"center": true}`, which subtracts it, and `geometry/span`
downstream.
""",
    inputs=(
        In("items", "activations/vector | logits/distribution | records/record",
           "The items to compare: any collection whose kind declares metrics.",
           many=True),
    ),
    output=(
        Output('geometry/similarity', collection=True, doc='One item per group: `{group, layer?, head?, space?, ids, labels, matrix, pairs?, separation?, nn_purity?, silhouette?}`, `labels` being the items\' values on the `axis` coordinate. The header carries `metric`, `metric_kind`, `symmetric`, `options`, `over` (the kind compared), `by`, `axis`, and `position`/`point` for vectors.')
    ),
    params=(
        P("metric", "string",
          "Which of the kind's metrics to apply. By default the kind's first: "
          "`cosine` for vectors, `jensen-shannon` for distributions, "
          "`hamming` for records.",
          None),
        P("options", "map[string, json]",
          "The metric's options, as it declares them — `{\"center\": true}` "
          "for `cosine`.",
          None),
        P("by", "string | null",
          "The header axis to group on before comparing: `\"space\"` (per "
          "layer and head), a coordinate name, or `null` for one group. "
          "Defaults to `\"space\"` when the items carry one.",
          None),
        P("axis", "string",
          "The coordinate the separation reads. `label` reads the older "
          "`label` field as well.",
          "label"),
    ),
    example={"metric": "cosine", "options": {"center": True}, "axis": "genre"},
    example_inputs={"items": {"$ref": {"bench": "you/lab/vectors"}}},
)


def run(ctx, inputs, params):
    return compare_geometry(inputs, params)


PAIRS_LIMIT = 32


def _group_key(item: Mapping[str, Any], by: str | None) -> tuple[str, dict[str, Any]]:
    if by in (None, "", "none"):
        return "all", {}
    if by == "space":
        layer, head = S.layer_of(item), S.head_of(item)
        fields: dict[str, Any] = {"layer": layer}
        name = f"layer={layer}"
        if head is not None:
            fields["head"] = head
            name += f",head={head}"
        return name, fields
    coords = S.coords_of(item)
    value = coords.get(by, item.get(by))
    return f"{by}={value}", {by: value}


def score_separation(matrix: np.ndarray, labels: list[Any], *, distance: bool) -> dict[str, Any]:
    n = len(labels)
    intra: list[float] = []
    inter: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels[i] == labels[j] else inter).append(float(matrix[i, j]))
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    gap = (inter_mean - intra_mean) if distance else (intra_mean - inter_mean)
    out: dict[str, Any] = {
        "separation": {"intra": round(intra_mean, 4), "inter": round(inter_mean, 4),
                       "gap": round(gap, 4)},
    }
    m = np.array(matrix, dtype=np.float64)
    np.fill_diagonal(m, np.inf if distance else -np.inf)
    nn = np.argmin(m, axis=1) if distance else np.argmax(m, axis=1)
    lab = np.asarray(labels, dtype=object)
    out["nn_purity"] = round(float((lab[nn] == lab).mean()), 4)
    with contextlib.suppress(Exception):
        from sklearn.metrics import silhouette_score

        d = np.array(matrix, dtype=np.float64) if distance else (1.0 - np.array(matrix, dtype=np.float64))
        d = np.clip(d, 0.0, None)
        np.fill_diagonal(d, 0.0)
        unique = list(dict.fromkeys(labels))
        ints = np.array([unique.index(lb) for lb in labels])
        out["silhouette"] = round(float(silhouette_score(d, ints, metric="precomputed")), 4)
    return out


def compare_geometry(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    src = inputs.get("items")
    if not isinstance(src, Mapping):
        raise ValueError("geometry/compare needs a collection on its `items` port")
    item_kind = K.item_kind_of(src)
    if item_kind is None:
        raise ValueError("geometry/compare needs a collection of a kind that declares metrics")
    items = K.items_of(src)
    metric_name = params.get("metric")
    _, metric, _fn = M.resolve(item_kind, metric_name)
    options = M.options_of(metric, params.get("options"))
    axis = str(params.get("axis") or "label")
    by = params.get("by", "space" if any("space" in it or "layer" in it for it in items) else None)
    distance = metric.kind == "distance"

    groups: dict[str, tuple[dict[str, Any], list[Mapping[str, Any]]]] = {}
    for it in items:
        name, fields = _group_key(it, by)
        groups.setdefault(name, (fields, []))[1].append(it)

    out: list[dict[str, Any]] = []
    for name, (fields, members) in groups.items():
        if len(members) < 2:
            continue
        mat, _m, _o = M.matrix(item_kind, members, metric.name, options)
        ids = [it.get("id") for it in members]
        labels = [S.label_of(it, axis) for it in members]
        entry: dict[str, Any] = {"group": name, **fields, "ids": ids, "labels": labels,
                                 "matrix": [[float(x) for x in row] for row in mat]}
        if "space" in members[0] or "layer" in members[0]:
            entry["space"] = S.space_of(members[0], header=src)
        if len(members) <= PAIRS_LIMIT:
            pairs = [{"a": ids[i], "b": ids[j], "value": round(float(mat[i, j]), 6)}
                     for i in range(len(ids)) for j in range(i + 1, len(ids))]
            entry["pairs"] = sorted(pairs, key=lambda p: (p["value"] if distance else -p["value"]))
        if all(lb is not None for lb in labels) and len(set(labels)) > 1:
            entry.update(score_separation(mat, labels, distance=distance))
        out.append(entry)
    if not out:
        raise ValueError("geometry/compare: no group has two items to compare")

    header: dict[str, Any] = {
        "metric": metric.name, "metric_kind": metric.kind, "symmetric": metric.symmetric,
        "options": options, "over": item_kind, "axis": axis, "by": by,
    }
    for k in ("position", "point"):
        if src.get(k) is not None:
            header[k] = src[k]
    return K.collection(
        "geometry/similarity", out, **header,
        description=(f"Pairwise {metric.name} over {item_kind} items"
                     + (f" grouped by {by}" if by else "")
                     + "; separation on the grouping axis where labels exist."))
