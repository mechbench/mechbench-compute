"""Directions as first-class objects (task 000367, epic 000364).

A `direction` is a unit vector at a (layer, point) with provenance: how
it was made (difference of means, PCA, a probe's weight, an SAE
feature, a trained steering vector, arithmetic over other directions),
from which objects, on which model. One object type flows through
`intervene` (add / project-out / rotate), probe-apply, attribution and
the vocabulary projection, so a direction found one way can be tried
every other way without conversion.

Producers here are pure (numpy over vector records) except
`vocab_projection`, which needs a model's unembedding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

KIND = "direction"


# --- construction ----------------------------------------------------------


def make(vector: Any, *, layer: int | None, point: str, method: str,
         sources: Sequence[str] = (), model: str | None = None,
         labels: Mapping[str, Any] | None = None,
         extra: Mapping[str, Any] | None = None,
         unit: bool = True) -> dict[str, Any]:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if unit:
        if norm == 0.0:
            raise ValueError("a zero vector cannot be a direction")
        v = v / norm
    prov: dict[str, Any] = {"method": method, "sources": list(sources),
                            "model": model}
    if labels:
        prov["labels"] = dict(labels)
    if extra:
        prov.update(dict(extra))
    return {
        "kind": KIND,
        "layer": None if layer is None else int(layer),
        "point": str(point),
        "d": int(v.size),
        "vector": [round(float(x), 6) for x in v],
        "norm": round(norm, 6),
        "unit": bool(unit),
        "provenance": prov,
    }


def as_array(d: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(d, Mapping) or d.get("kind") != KIND:
        raise ValueError("expected a direction object (kind 'direction')")
    return np.asarray(d["vector"], dtype=np.float32).reshape(-1)


def same_space(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    for k in ("layer", "point", "d"):
        if a.get(k) != b.get(k):
            raise ValueError(
                f"directions live in different spaces: {k} {a.get(k)!r} vs {b.get(k)!r}")


def _point_of(vectors: Mapping[str, Any], override: str | None) -> str:
    if override:
        return override
    p = str(vectors.get("point", "post"))
    return p if "." in p or p.startswith("resid_") else f"resid_{p}"


def _rows_at(vectors: Mapping[str, Any], layer: int) -> list[Mapping[str, Any]]:
    if not isinstance(vectors, Mapping) or vectors.get("kind") != "residual_vectors":
        raise ValueError("expected a residual_vectors record")
    rows = [r for r in vectors.get("rows", []) if r.get("layer") == layer]
    if not rows:
        raise ValueError(f"the vectors record has no rows at layer {layer}")
    return rows


# --- producers --------------------------------------------------------------


def from_vectors(vectors: Mapping[str, Any], *, layer: int, positive: str,
                 negative: str, point: str | None = None,
                 source: str | None = None) -> dict[str, Any]:
    """Difference of means: centroid(positive) − centroid(negative) at
    `layer`, over a residual_vectors record whose rows carry `label`."""
    rows = _rows_at(vectors, layer)
    pos = np.array([r["vector"] for r in rows if r.get("label") == positive],
                   dtype=np.float32)
    neg = np.array([r["vector"] for r in rows if r.get("label") == negative],
                   dtype=np.float32)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            f"no rows at layer {layer} for labels {positive!r}/{negative!r}")
    return make(pos.mean(0) - neg.mean(0), layer=layer,
                point=_point_of(vectors, point), method="diff_of_means",
                sources=[source] if source else [],
                model=vectors.get("model"),
                labels={"positive": positive, "negative": negative},
                extra={"n_positive": len(pos), "n_negative": len(neg)})


def from_pca(vectors: Mapping[str, Any], *, layer: int, component: int = 0,
             label: str | None = None, point: str | None = None,
             source: str | None = None) -> dict[str, Any]:
    """A principal component of the (centered) rows at `layer`, optionally
    restricted to one label. Sign is fixed so the largest-magnitude
    coordinate is positive (a component has no intrinsic sign)."""
    rows = _rows_at(vectors, layer)
    if label is not None:
        rows = [r for r in rows if r.get("label") == label]
    x = np.array([r["vector"] for r in rows], dtype=np.float32)
    if len(x) < 2:
        raise ValueError("PCA needs at least two rows")
    x = x - x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    if component >= len(s):
        raise ValueError(f"component {component} out of range ({len(s)} available)")
    v = vt[component]
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    explained = float(s[component] ** 2 / max(float((s ** 2).sum()), 1e-12))
    return make(v, layer=layer, point=_point_of(vectors, point), method="pca",
                sources=[source] if source else [], model=vectors.get("model"),
                labels={"label": label} if label else None,
                extra={"component": int(component), "explained": round(explained, 4),
                       "n_rows": len(x)})


# --- arithmetic (pure) --------------------------------------------------------


def add(directions: Sequence[Mapping[str, Any]],
        weights: Sequence[float] | None = None) -> dict[str, Any]:
    if not directions:
        raise ValueError("add needs at least one direction")
    ws = [1.0] * len(directions) if weights is None else [float(w) for w in weights]
    if len(ws) != len(directions):
        raise ValueError("weights must match directions")
    for d in directions[1:]:
        same_space(directions[0], d)
    v = sum(w * as_array(d) for w, d in zip(ws, directions, strict=True))
    return make(v, layer=directions[0]["layer"], point=directions[0]["point"],
                method="add", sources=[str(d.get("provenance", {}).get("method"))
                                       for d in directions],
                model=directions[0].get("provenance", {}).get("model"),
                extra={"weights": ws})


def average(directions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = add(directions)
    out["provenance"]["method"] = "average"
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
    return make(v, layer=d["layer"], point=d["point"], method="orthogonalize",
                model=d.get("provenance", {}).get("model"),
                extra={"against": len(basis)})


def normalize(d: Mapping[str, Any]) -> dict[str, Any]:
    return make(as_array(d), layer=d["layer"], point=d["point"], method="normalize",
                model=d.get("provenance", {}).get("model"))


def similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    same_space(a, b)
    va, vb = as_array(a), as_array(b)
    cos = float(va @ vb / max(float(np.linalg.norm(va) * np.linalg.norm(vb)), 1e-12))
    return {"kind": "direction_similarity", "cosine": round(cos, 6),
            "layer": a["layer"], "point": a["point"]}


def project_rows(vectors: Mapping[str, Any], d: Mapping[str, Any]) -> dict[str, Any]:
    """Each row of a residual_vectors record at the direction's layer,
    projected onto the direction: the scalar coordinate along it."""
    rows = _rows_at(vectors, int(d["layer"]))
    u = as_array(d)
    out = []
    for r in rows:
        v = np.asarray(r["vector"], dtype=np.float32)
        if v.size != u.size:
            raise ValueError("vector width does not match the direction")
        out.append({k: r[k] for k in r if k != "vector"} | {"projection": round(float(v @ u), 5)})
    return {"kind": "projections", "layer": d["layer"], "point": d["point"], "rows": out}


# --- the unembedding as a lens -------------------------------------------------------


def vocab_projection(model, d: Mapping[str, Any], *, top_k: int = 10) -> dict[str, Any]:
    """What a direction 'says' in token space: the top tokens of the
    unembedding applied to +d and to −d (the final norm is scale-
    invariant, so a unit direction is as good as any multiple)."""
    u = as_array(d)
    out: dict[str, Any] = {"kind": "direction_vocab", "layer": d["layer"],
                           "point": d["point"], "top_k": int(top_k)}
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        probs = model.decoded_distribution(sign * u)
        order = np.argsort(-probs)[:top_k]
        out[name] = [{"token": model.tokenizer.decode([int(t)]),
                      "p": round(float(probs[int(t)]), 5)} for t in order]
    return out


# --- pure-block adapters (inputs, params) ---------------------------------------------


def _directions_from(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    got: list[Mapping[str, Any]] = []
    listed = inputs.get("directions") or params.get("directions")
    if isinstance(listed, list):
        got.extend(listed)
    for k in sorted(inputs):
        v = inputs[k]
        if isinstance(v, Mapping) and v.get("kind") == KIND and k != "directions":
            got.append(v)
    if not got:
        raise ValueError("no direction objects on the inputs (ports) or params.directions")
    return got


def block_from_vectors(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    vectors = inputs.get("vectors") or params.get("vectors")
    return from_vectors(vectors, layer=int(params["layer"]),
                        positive=str(params["positive"]), negative=str(params["negative"]),
                        point=params.get("point"), source=params.get("source"))


def block_from_pca(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    vectors = inputs.get("vectors") or params.get("vectors")
    return from_pca(vectors, layer=int(params["layer"]),
                    component=int(params.get("component", 0)),
                    label=params.get("label"), point=params.get("point"),
                    source=params.get("source"))


def block_add(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return add(_directions_from(inputs, params), params.get("weights"))


def block_average(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return average(_directions_from(inputs, params))


def block_orthogonalize(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    d = inputs.get("direction") or params.get("direction")
    against = inputs.get("against") or params.get("against")
    if isinstance(against, Mapping):
        against = [against]
    return orthogonalize(d, list(against or []))


def block_normalize(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return normalize(inputs.get("direction") or params.get("direction"))


def block_similarity(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    a = inputs.get("a") or params.get("a")
    b = inputs.get("b") or params.get("b")
    return similarity(a, b)


def block_project(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return project_rows(inputs.get("vectors") or params.get("vectors"),
                        inputs.get("direction") or params.get("direction"))


PURE_DIRECTION_BLOCKS = {
    "~canonical/ops/direction/from-vectors/1": block_from_vectors,
    "~canonical/ops/direction/from-pca/1": block_from_pca,
    "~canonical/ops/direction/add/1": block_add,
    "~canonical/ops/direction/average/1": block_average,
    "~canonical/ops/direction/orthogonalize/1": block_orthogonalize,
    "~canonical/ops/direction/normalize/1": block_normalize,
    "~canonical/ops/direction/similarity/1": block_similarity,
    "~canonical/ops/direction/project/1": block_project,
}
