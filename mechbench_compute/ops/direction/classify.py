from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points as hookpoints
from mechbench_compute import shapes as S
from mechbench_compute.directions.constants import DEFAULT_AXIS
from mechbench_compute.directions.make import make
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="direction/classify",
    summary=(
        "Fit a linear probe at every layer — the direction that separates "
        "one label from the rest — and report how much of it a held-out "
        "item shows, over the majority baseline."
    ),
    description="""\
`direction/fit` takes the difference of two centroids: an answer that
always exists and never says how good it is. A probe is the same question
asked so that it can be wrong. The items at each space are split, a
logistic boundary is fitted on one part, and its accuracy is read on the
part it never saw — beside `baseline`, the share of the commonest label,
which is what answering without looking would score.

Run at every layer, the accuracies are a curve: WHERE a distinction
becomes linearly decodable, which is the standard probing result and the
thing one difference of centroids cannot produce. Plot it with
`records/plot x: "layer", y: "accuracy_test"`, or find the layer with
`records/rank value: "accuracy_test", k: 1`.

Two labels give one direction per space, pointing from the negative label
towards the positive. More give one per label, each against the rest
(`negative: "rest"`), all sharing the space's accuracy and each with its
own `auc`. Every item is an ordinary `direction/vector`, so the probe that
decodes best is the direction to steer along or project out — select it
and wire it into `intervene/apply`.

A probe needs at least eight labelled items at a space and both labels on
each side of the split; it refuses by name rather than reporting a number
nobody should read.
""",
    inputs=(
        In("vectors", "activations/vector",
           "Labelled vectors: items carrying the `axis` value among their "
           "coordinates, at one or more spaces.", many=True),
    ),
    output=Output('direction/vector', collection=True,
                  doc='One item per space per label: a direction with `coords` (`layer`, `label`, `head` when heads differ), the headline scores at the top level (`accuracy_test`, `accuracy_train`, `baseline`, `over_baseline`, `confidence_test`, `auc`, `n_items`, `n_train`, `n_test`) so a table and a chart read them, and the whole fit in `derivation` (`method: "logistic"`, `positive`, `negative`, `classes`, `C`, `seed`). The header carries `axis`, `method`, `holdout` and `seed`.'),
    params=(
        P("axis", "string",
          "The coordinate holding the label to separate. Items without one "
          "are left out.",
          "label"),
        P("layers", "list[int]",
          "Which layers to probe. Every layer the items carry, by default.",
          None),
        P("holdout", "float",
          "The fraction of items held out of each fit, to score it.", 0.2),
        P("C", "float",
          "The inverse regularisation strength of the logistic fit: smaller "
          "is a stronger prior that the boundary is simple.",
          1.0),
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
    example={"axis": "sense", "holdout": 0.25},
    example_inputs={"vectors": {"$ref": {"bench": "you/lab/residuals"}}},
)


def run(ctx, inputs, params):
    return fit_probe(inputs.get("vectors"),
                     axis=str(params.get("axis") or DEFAULT_AXIS),
                     layers=params.get("layers"),
                     holdout=float(params.get("holdout", 0.2)),
                     seed=int(params.get("seed", 0)),
                     C=float(params.get("C", 1.0)),
                     point=params.get("point"), source=params.get("source"))


def fit_probe(vectors: Mapping[str, Any], *, axis: str = DEFAULT_AXIS,
              layers: Sequence[int] | None = None, holdout: float = 0.2,
              seed: int = 0, C: float = 1.0, point: str | None = None,
              source: str | None = None) -> dict[str, Any]:
    """A linear probe per space: which way the items of one label lie
    from the rest, and how much of that a held-out item shows.

    `direction/fit` takes the difference of two centroids — an answer
    that always exists and never says how good it is. A probe is the
    same question asked so that it can be wrong: fit a boundary on part
    of the items, and read its accuracy on the part it never saw, over
    the majority-class baseline. Run at every layer, the accuracies are
    the curve that says WHERE something becomes linearly decodable,
    which is the standard probing result and the thing a single
    difference of centroids cannot produce.

    One item per (space × label): a `direction/vector` whose derivation
    carries the fit, and whose headline scores sit at the top level so
    `records/plot x: "layer", y: "accuracy_test"` is the figure and
    `records/rank` finds the best layer. Two labels give one direction
    (from the negative label towards the positive); more give one per
    label, each against the rest.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from mechbench_compute.lexicon import kinds as K

    if not isinstance(vectors, Mapping) or K.item_kind_of(vectors) != "activations/vector":
        raise ValueError("expected a collection of activations/vector")
    wanted = None if layers is None else {int(x) for x in layers}
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for r in K.items_of(vectors):
        sp = S.space_of(r, header=vectors)
        if wanted is not None and sp.get("layer") not in wanted:
            continue
        if S.label_of(r, axis) is None:
            continue
        groups.setdefault((sp.get("layer"), sp.get("point"), sp.get("head")), []).append(r)
    if not groups:
        raise ValueError(
            f"no items carry {axis!r}" + ("" if layers is None else f" at layers {sorted(wanted or [])}"))

    rng = np.random.default_rng(int(seed))
    items: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(-1 if x is None else x for x in k)):
        rows = groups[key]
        labels = [S.label_of(r, axis) for r in rows]
        classes = sorted({str(v) for v in labels}, key=str)
        if len(classes) < 2:
            raise ValueError(
                f"a probe needs two labels on {axis!r}; layer {key[0]} has "
                f"only {classes[0]!r}")
        if len(rows) < 8:
            raise ValueError(
                f"a probe needs at least 8 labelled items; layer {key[0]} has {len(rows)}")
        x = np.array([r["vector"] for r in rows], dtype=np.float32)
        y = np.array([str(v) for v in labels])
        order = rng.permutation(len(rows))
        n_test = max(1, int(round(len(rows) * float(holdout))))
        test, train = order[:n_test], order[n_test:]
        if len(set(y[train])) < 2:
            raise ValueError(
                f"the split leaves one label at layer {key[0]}: hold out less, "
                "or label more items")
        fit = LogisticRegression(C=float(C), max_iter=2000).fit(x[train], y[train])
        pred_test, pred_train = fit.predict(x[test]), fit.predict(x[train])
        proba = fit.predict_proba(x[test])
        accuracy_test = float(np.mean(pred_test == y[test]))
        # What a probe has to beat: always answering the commonest label.
        counts = {c: int(np.sum(y[test] == c)) for c in classes}
        baseline = float(max(counts.values()) / len(test))
        scores = {
            "accuracy_test": round(accuracy_test, 4),
            "accuracy_train": round(float(np.mean(pred_train == y[train])), 4),
            "baseline": round(baseline, 4),
            "over_baseline": round(accuracy_test - baseline, 4),
            # How sure it is when it answers — beside the accuracy, this
            # is how a confidently wrong probe shows itself.
            "confidence_test": round(float(np.mean(np.max(proba, axis=1))), 4),
            "n_items": len(rows), "n_train": int(len(train)), "n_test": int(len(test)),
        }
        sp = S.space_of(rows[0], header=vectors)
        if point:
            sp["point"] = hookpoints.normalize(str(point))
        classes_fit = [str(c) for c in fit.classes_]
        binary = len(classes_fit) == 2
        for i, cls in enumerate(classes_fit):
            if binary and i == 0:
                continue        # one boundary: the direction points at classes_[1]
            coef = fit.coef_[0] if binary else fit.coef_[i]
            if not np.any(coef):
                raise ValueError(
                    f"the probe for {cls!r} at layer {key[0]} is a zero vector — "
                    "the labels are not linearly separated at all here")
            in_test = y[test] == cls
            auc = (round(float(roc_auc_score(in_test, proba[:, i])), 4)
                   if 0 < int(in_test.sum()) < len(test) else None)
            negative = classes_fit[0] if binary else "rest"
            item = make(coef, dict(sp), method="logistic",
                        sources=[source] if source else [],
                        labels={"axis": axis, "positive": cls, "negative": negative},
                        extra={**scores, "auc": auc, "C": float(C), "seed": int(seed),
                               "classes": classes_fit})
            item["id"] = f"L{key[0]}:{cls}" if key[2] is None else f"L{key[0]}h{key[2]}:{cls}"
            item["coords"] = {"layer": key[0], "label": cls,
                              **({"head": key[2]} if key[2] is not None else {})}
            # The headline numbers where a table and a chart read them.
            item.update({k: v for k, v in scores.items()})
            if auc is not None:
                item["auc"] = auc
            items.append(item)
    return K.collection("direction/vector", items, axis=axis,
                        method="logistic", holdout=float(holdout), seed=int(seed),
                        description=(
                            f"A linear probe for {axis!r} at each space: the direction "
                            "that separates a label from the rest, and how much of it "
                            "a held-out item shows."))

