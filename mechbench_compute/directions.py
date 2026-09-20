"""Directions as first-class objects (task 000367, epic 000364).

A `direction/vector` is a unit vector in a model's activation space
with provenance: how it was made (difference of means, PCA, a probe's
weight, an SAE feature, a trained steering vector, arithmetic over
other directions), from which objects, on which model. It is an
`activations/vector` — `{space, vector, norm}` — plus `unit` and
`derivation`, so one object type flows through `intervene/apply`
(add / project-out / rotate), `direction/project`, attribution and the
vocabulary projection, and a direction found one way can be tried every
other way without conversion.

Producers here are pure (numpy over vector items) except
`vocab_projection`, which needs a model's unembedding.
"""

from __future__ import annotations

import hashlib

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import points as P
from mechbench_compute import shapes as S

KIND = "direction/vector"

#: The coordinate the grouping ops read when none is named: the retired
#: `label` field is read as a `label` coordinate, so older vector
#: collections group as they did.
DEFAULT_AXIS = "label"


def _is_direction(d: Any) -> bool:
    """A direction record, by its current name or the retired one."""
    from mechbench_compute.lexicon import kinds as K

    if not isinstance(d, Mapping) or not isinstance(d.get("kind"), str):
        return False
    try:
        return K.resolve_kind(d["kind"])[0] == KIND
    except KeyError:
        return False


# --- construction ----------------------------------------------------------


def make(vector: Any, sp: Mapping[str, Any], *, method: str,
         sources: Sequence[str] = (), labels: Mapping[str, Any] | None = None,
         extra: Mapping[str, Any] | None = None, unit: bool = True) -> dict[str, Any]:
    """A direction in space `sp`. `labels` is the grouping it was built
    from (`{axis, positive, negative}`); `extra` rides in the derivation."""
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if unit:
        if norm == 0.0:
            raise ValueError("a zero vector cannot be a direction")
        v = v / norm
    prov: dict[str, Any] = {"method": method, "sources": list(sources),
                            "model": sp.get("model")}
    if labels:
        prov.update({k: v_ for k, v_ in labels.items() if v_ is not None})
    if extra:
        prov.update(dict(extra))
    # `derivation`, not `provenance`: the latter is the Emitted envelope's
    # field, and bench.emit refuses to wrap a payload that carries one.
    item = S.vector(v, sp)
    item["norm"] = round(norm, 6)
    item["unit"] = bool(unit)
    item["derivation"] = prov
    return {"kind": KIND, **item}


def as_array(d: Mapping[str, Any]) -> np.ndarray:
    if not _is_direction(d):
        raise ValueError("expected a direction object (kind 'direction/vector')")
    return np.asarray(d["vector"], dtype=np.float32).reshape(-1)


def same_space(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    S.same_space(a, b, what="directions")


def space_of(d: Mapping[str, Any]) -> dict[str, Any]:
    return S.space_of(d)


def _items_at(vectors: Mapping[str, Any], layer: int) -> list[Mapping[str, Any]]:
    from mechbench_compute.lexicon import kinds as K

    if not isinstance(vectors, Mapping) or K.item_kind_of(vectors) != "activations/vector":
        raise ValueError("expected a collection of activations/vector")
    rows = [r for r in K.items_of(vectors) if S.layer_of(r) == layer]
    if not rows:
        raise ValueError(f"the vectors collection has no items at layer {layer}")
    return rows


def _models_of(vectors: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The distinct models the rows were captured from, in order."""
    seen: dict[str, None] = {}
    for r in rows:
        m = S.space_of(r, header=vectors).get("model")
        if m is not None:
            seen[str(m)] = None
    return list(seen)


def _space_at(vectors: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
              point: str | None) -> dict[str, Any]:
    """The rows' shared space, with the point overridden when asked.

    Rows from more than one model — a base and an adapted capture in one
    union — give a direction that lives in neither: its `model` is None
    (the residual basis they share), and the derivation names them all,
    so two such axes compare and the models are still on record."""
    sp = S.space_of(rows[0], header=vectors)
    if len(_models_of(vectors, rows)) > 1:
        sp["model"] = None
    if point:
        sp["point"] = P.normalize(str(point))
    sp["head"] = None
    return sp


def _model_provenance(vectors: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    models = _models_of(vectors, rows)
    return {"models": models} if len(models) > 1 else {}


# --- producers --------------------------------------------------------------


def from_vectors(vectors: Mapping[str, Any], *, layer: int, positive: str,
                 negative: str, axis: str = DEFAULT_AXIS, point: str | None = None,
                 source: str | None = None) -> dict[str, Any]:
    """Difference of means: centroid(`positive`) − centroid(`negative`)
    at `layer`, the groups being the items' values on the `axis`
    coordinate."""
    rows = _items_at(vectors, layer)
    pos = np.array([r["vector"] for r in rows if str(S.label_of(r, axis)) == str(positive)],
                   dtype=np.float32)
    neg = np.array([r["vector"] for r in rows if str(S.label_of(r, axis)) == str(negative)],
                   dtype=np.float32)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            f"no items at layer {layer} with {axis}={positive!r}/{negative!r}")
    return make(pos.mean(0) - neg.mean(0), _space_at(vectors, rows, point),
                method="diff_of_means", sources=[source] if source else [],
                labels={"axis": axis, "positive": positive, "negative": negative},
                extra={"n_positive": len(pos), "n_negative": len(neg),
                       **_model_provenance(vectors, rows)})


def from_regression(vectors: Mapping[str, Any], *, layer: int, target: str,
                    alphas: Sequence[float] | None = None, holdout: float = 0.2,
                    seed: int = 0, point: str | None = None,
                    source: str | None = None) -> dict[str, Any]:
    """Ridge regression of the items' vectors against a continuous
    coordinate; the fitted weight vector IS the direction (task 000586).

    `from_vectors` answers "which way does THIS group lie from THAT
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
            if S.layer_of(r) != layer or _number_at(r, target) is None:
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
        y = float(_number_at(r, target) or 0.0)
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
        y = float(_number_at(r, target) or 0.0)
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
        sp["point"] = P.normalize(str(point))
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


def _number_at(row: Mapping[str, Any], name: str) -> float | None:
    """The row's value for `name`, from its coordinates or its top level,
    when that value is a number. None when it is absent or is not one."""
    coords = row.get("coords")
    v = (coords.get(name) if isinstance(coords, Mapping) else None)
    if v is None:
        v = row.get(name)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


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


# --- arithmetic (pure) --------------------------------------------------------


def add(directions: Sequence[Mapping[str, Any]],
        weights: Sequence[float] | None = None) -> dict[str, Any]:
    """Weighted sum of directions in one space, re-normalized.

    The composition primitive: steering along "formal" and "terse" at
    once is their sum, and `weights` sets the mix. Every input must share
    a space — adding across spaces is meaningless, and `same_space`
    refuses it rather than returning a plausible vector.
    """
    if not directions:
        raise ValueError("add needs at least one direction")
    ws = [1.0] * len(directions) if weights is None else [float(w) for w in weights]
    if len(ws) != len(directions):
        raise ValueError("weights must match directions")
    for d in directions[1:]:
        same_space(directions[0], d)
    v = sum(w * as_array(d) for w, d in zip(ws, directions, strict=True))
    return make(v, space_of(directions[0]), method="add",
                sources=[str(d.get("derivation", {}).get("method")) for d in directions],
                extra={"weights": ws})


def average(directions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The equal-weight mean of several directions: the shared component.

    Distinct from `add` only in intent and in what the derivation
    records, since both normalize — but the question "what do these
    adapters have in common?" is answered by the mean of their UNIT
    directions, which weights each one equally however long it is.
    """
    out = add(directions)
    out["derivation"]["method"] = "average"
    return out


def orthogonalize(d: Mapping[str, Any],
                  against: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Remove from `d` its components along each of `against`
    (Gram–Schmidt against an orthonormalized basis of them)."""
    v = as_array(d)
    basis: list[np.ndarray] = []
    for a in against:
        same_space(d, a)
        u = as_array(a)
        for b in basis:
            u = u - float(u @ b) * b
        n = float(np.linalg.norm(u))
        if n > 1e-8:
            basis.append(u / n)
    for b in basis:
        v = v - float(v @ b) * b
    if float(np.linalg.norm(v)) < 1e-8:
        raise ValueError("direction lies entirely in the span of `against`")
    return make(v, space_of(d), method="orthogonalize", extra={"against": len(basis)})


def normalize(d: Mapping[str, Any]) -> dict[str, Any]:
    """A direction rescaled to unit length, keeping its space and model.

    Directions are stored unit already, so this is for the case where one
    arrived otherwise — hand-built, or read from an external source —
    and for making the normalization an explicit, recorded step rather
    than an implicit one.
    """
    return make(as_array(d), space_of(d), method="normalize")


def project_rows(vectors: Mapping[str, Any], d: Mapping[str, Any]) -> dict[str, Any]:
    """Each item of a vector collection at the direction's layer,
    projected onto the direction: the scalar coordinate along it."""
    layer = S.layer_of(d)
    rows = _items_at(vectors, layer)
    u = as_array(d)
    out = []
    for r in rows:
        v = np.asarray(r["vector"], dtype=np.float32)
        if v.size != u.size:
            raise ValueError("vector width does not match the direction")
        out.append(S.coordinate(float(v @ u), S.space_of(r, header=vectors), d,
                                id=r.get("id"), coords=S.coords_of(r),
                                token=r.get("token")))
    from mechbench_compute.lexicon import kinds as K

    return K.collection("activations/coordinate", out, projected=True)


# --- the unembedding as a lens -------------------------------------------------------


def vocab_projection(model, d: Mapping[str, Any], *, top_k: int = 10) -> dict[str, Any]:
    """What a direction 'says' in token space: the distribution the
    unembedding gives +d and −d (the final norm is scale-invariant, so a
    unit direction is as good as any multiple)."""
    u = as_array(d)
    out: dict[str, Any] = {"kind": "direction/vocab", "space": space_of(d), "top_k": int(top_k)}
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        probs = np.asarray(model.decoded_distribution(sign * u), dtype=np.float64)
        logp = np.log(np.clip(probs, 1e-300, None))
        out[name] = S.distribution(logp, model.tokenizer, top_k=top_k)
    return out


# --- pure-block adapters (inputs, params) ---------------------------------------------


def _directions_from(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    got: list[Mapping[str, Any]] = []
    listed = inputs.get("directions")
    if isinstance(listed, list):
        got.extend(listed)
    for k in sorted(inputs):
        v = inputs[k]
        if _is_direction(v) and k != "directions":
            got.append(v)
    if not got:
        raise ValueError("no direction objects on the inputs (ports) or params.directions")
    return got


def block_from_vectors(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    vectors = inputs.get("vectors")
    return from_vectors(vectors, layer=int(params["layer"]),
                        axis=str(params.get("axis") or DEFAULT_AXIS),
                        positive=str(params["positive"]), negative=str(params["negative"]),
                        point=params.get("point"), source=params.get("source"))


def block_from_regression(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return from_regression(inputs.get("vectors"), layer=int(params["layer"]),
                           target=str(params["target"]),
                           alphas=params.get("alphas"),
                           holdout=float(params.get("holdout", 0.2)),
                           seed=int(params.get("seed", 0)),
                           point=params.get("point"), source=params.get("source"))


def block_from_pca(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    vectors = inputs.get("vectors")
    # `label` is the retired spelling of `value` on the `label` axis.
    value = params.get("value", params.get("label"))
    return from_pca(vectors, layer=int(params["layer"]),
                    component=int(params.get("component", 0)),
                    axis=str(params.get("axis") or DEFAULT_AXIS), value=value,
                    point=params.get("point"), source=params.get("source"))


def block_add(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return add(_directions_from(inputs, params), params.get("weights"))


def block_average(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return average(_directions_from(inputs, params))


def block_orthogonalize(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    d = inputs.get("direction")
    against = inputs.get("against")
    if isinstance(against, Mapping):
        against = [against]
    return orthogonalize(d, list(against or []))


def block_normalize(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return normalize(inputs.get("direction"))


def block_project(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return project_rows(inputs.get("vectors"),
                        inputs.get("direction"))


PURE_DIRECTION_BLOCKS = {
    "direction/fit": block_from_vectors,
    "direction/regress": block_from_regression,
    "direction/decompose": block_from_pca,
    "direction/add": block_add,
    "direction/average": block_average,
    "direction/orthogonalize": block_orthogonalize,
    "direction/normalize": block_normalize,
    "direction/project": block_project,
}
