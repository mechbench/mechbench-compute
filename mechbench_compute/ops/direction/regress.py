from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points as hookpoints
from mechbench_compute import shapes as S
from mechbench_compute.directions.make import make
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.lexicon.direction import _point, _source

OP = Op(
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


def run(ctx, inputs, params):
    return fit_regression(inputs.get("vectors"), layer=int(params["layer"]),
                          target=str(params["target"]),
                          alphas=params.get("alphas"),
                          holdout=float(params.get("holdout", 0.2)),
                          seed=int(params.get("seed", 0)),
                          point=params.get("point"), source=params.get("source"))


def fit_regression(vectors: Mapping[str, Any], *, layer: int, target: str,
                   alphas: Sequence[float] | None = None, holdout: float = 0.2,
                   seed: int = 0, point: str | None = None,
                   source: str | None = None) -> dict[str, Any]:
    """Ridge regression of the items' vectors against a continuous
    coordinate; the fitted weight vector IS the direction (task 000586).

    `fit_mean_difference` answers "which way does THIS group lie from THAT
    one" — two labels and a difference of centroids. Some signals are
    not two groups: a token's surprisal, a passage's length, a score.
    For those the question is which way the residual moves as the
    quantity rises, and the answer is a regression, not a contrast.

    The fit holds out a fixed fraction so the derivation can say how
    much of the signal the direction actually carries: a weight vector
    always exists, and R² on unseen items is what says whether it means
    anything. The alpha is the one that does best on the holdout.

    The rows are read in two passes and never held whole (000613): the
    first accumulates the training normal equations (d × d, with an
    unpenalised intercept), which are solved for every alpha at once;
    the second scores every alpha on the held-out rows. A collection of
    a million tokens costs the same memory as one of a thousand.
    """
    from mechbench_compute.lexicon import kinds as K

    if not isinstance(vectors, Mapping) or K.item_kind_of(vectors) != "activations/vector":
        raise ValueError("expected a collection of activations/vector")
    grid = [float(a) for a in (alphas or [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6])]
    frac = float(holdout)

    def rows_at_layer():
        """(index among kept rows, row) for the rows at `layer` carrying
        the target — the same rows in the same order on both passes."""
        i = 0
        for r in K.items_of(vectors):
            if S.layer_of(r) != layer or _read_number(r, target) is None:
                continue
            yield i, r
            i += 1

    def is_test(i: int) -> bool:
        # A deterministic holdout under the seed, decided per row from
        # its index — so neither pass needs the count in advance.
        h = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        return int.from_bytes(h[:4], "little") / 2**32 < frac

    # Pass one: the training normal equations, and the rows' lineage.
    xtx = xty = None
    n_train = n_test = n_at_layer = 0
    first: list[Mapping[str, Any]] = []
    models: dict[str, None] = {}
    for i, r in rows_at_layer():
        n_at_layer += 1
        if len(first) < 4:
            first.append(r)
        m = S.space_of(r, header=vectors).get("model")
        if m:
            models[str(m)] = None
        if is_test(i):
            n_test += 1
            continue
        x = np.append(np.asarray(r["vector"], dtype=np.float64), 1.0)
        y = float(_read_number(r, target) or 0.0)
        if xtx is None:
            xtx = np.zeros((x.size, x.size)); xty = np.zeros(x.size)
        xtx += np.outer(x, x)
        xty += x * y
        n_train += 1
    kept = n_train + n_test
    if kept < 8:
        raise ValueError(
            f"regression needs at least 8 items at layer {layer} carrying "
            f"{target!r}; {kept} have it")
    if n_train < 2 or n_test < 1:
        raise ValueError("holdout leaves too few items to fit")
    assert xtx is not None and xty is not None
    d = xtx.shape[0] - 1
    penalty = np.eye(d + 1); penalty[d, d] = 0.0      # the intercept is free
    weights = np.stack([np.linalg.solve(xtx + a * penalty, xty) for a in grid], axis=1)  # [d+1, k]

    # Pass two: every alpha scored on both sides at once.
    sse_test = np.zeros(len(grid)); sse_train = np.zeros(len(grid))
    sum_y_test = sum_y2_test = sum_y_train = sum_y2_train = 0.0
    sum_p_test = np.zeros(len(grid)); sum_py_test = np.zeros(len(grid)); sum_p2_test = np.zeros(len(grid))
    for i, r in rows_at_layer():
        x = np.append(np.asarray(r["vector"], dtype=np.float64), 1.0)
        y = float(_read_number(r, target) or 0.0)
        pred = x @ weights                                   # [k]
        if is_test(i):
            sse_test += (y - pred) ** 2
            sum_y_test += y; sum_y2_test += y * y
            sum_p_test += pred; sum_py_test += pred * y; sum_p2_test += pred * pred
        else:
            sse_train += (y - pred) ** 2
            sum_y_train += y; sum_y2_train += y * y

    def r2(sse, sy, sy2, n):
        sst = max(sy2 - sy * sy / n, 1e-12)
        return 1.0 - sse / sst

    r2_tests = r2(sse_test, sum_y_test, sum_y2_test, n_test)
    best = int(np.argmax(r2_tests))
    r2_trains = r2(sse_train, sum_y_train, sum_y2_train, n_train)
    # Pearson between prediction and truth on the holdout, from sums.
    cov = sum_py_test[best] / n_test - (sum_p_test[best] / n_test) * (sum_y_test / n_test)
    var_p = sum_p2_test[best] / n_test - (sum_p_test[best] / n_test) ** 2
    var_y = sum_y2_test / n_test - (sum_y_test / n_test) ** 2
    pearson = float(cov / np.sqrt(var_p * var_y)) if n_test > 1 and var_p > 0 and var_y > 0 else 0.0

    sp = S.space_of(first[0], header=vectors)
    if len(models) > 1:
        sp["model"] = None
    if point:
        sp["point"] = hookpoints.normalize(str(point))
    sp["head"] = None
    return make(np.asarray(weights[:d, best], dtype=np.float32), sp,
                method="ridge", sources=[source] if source else [],
                labels={"target": target},
                extra={"alpha": grid[best],
                       "r2_train": round(float(r2_trains[best]), 4),
                       "r2_test": round(float(r2_tests[best]), 4),
                       "pearson_test": round(pearson, 4),
                       "n_items": int(kept), "n_train": int(n_train),
                       "n_test": int(n_test), "seed": int(seed),
                       **({"models": list(models)} if len(models) > 1 else {})})


def _read_number(row: Mapping[str, Any], name: str) -> float | None:
    """The row's value for `name`, from its coordinates or its top level,
    when that value is a number. None when it is absent or is not one."""
    coords = row.get("coords")
    v = (coords.get(name) if isinstance(coords, Mapping) else None)
    if v is None:
        v = row.get(name)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)

