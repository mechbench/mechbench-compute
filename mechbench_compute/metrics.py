from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon._base import Metric

IMPLEMENTATIONS: dict[tuple[str, str], Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], np.ndarray]] = {}


def _implements(kind: str, name: str):
    def deco(fn):
        IMPLEMENTATIONS[(kind, name)] = fn
        return fn
    return deco


def declared(item_kind: str) -> dict[str, Metric]:
    out: dict[str, Metric] = {}
    for name in reversed(K.ancestry(item_kind)):
        for m in K.BY_KIND[name].metrics:
            out[m.name] = m
    return out


def default_metric(item_kind: str) -> str | None:
    for name in K.ancestry(item_kind):
        if K.BY_KIND[name].metrics:
            return K.BY_KIND[name].metrics[0].name
    return None


def resolve(item_kind: str, name: str | None) -> tuple[str, Metric, Callable]:
    name = name or default_metric(item_kind)
    if name is None:
        raise ValueError(f"{item_kind} declares no metrics — nothing to compare its items by")
    for kind in K.ancestry(item_kind):
        for m in K.BY_KIND[kind].metrics:
            if m.name == name:
                fn = IMPLEMENTATIONS.get((kind, name))
                if fn is None:
                    raise ValueError(f"metric {name!r} is declared on {kind} but not implemented")
                return kind, m, fn
    have = sorted(declared(item_kind))
    raise ValueError(
        f"{item_kind} has no metric {name!r}; it compares by {have}")


def options_of(metric: Metric, options: Mapping[str, Any] | None) -> dict[str, Any]:
    given = dict(options or {})
    known = {o.name: o for o in metric.options}
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(
            f"metric {metric.name!r} takes no option {', '.join(repr(u) for u in unknown)}; "
            f"its options: {sorted(known) or 'none'}")
    out = {o.name: o.default for o in metric.options if not o.required}
    out.update(given)
    return out


def matrix(item_kind: str, items: Sequence[Mapping[str, Any]], name: str | None = None,
           options: Mapping[str, Any] | None = None) -> tuple[np.ndarray, Metric, dict[str, Any]]:
    _, metric, fn = resolve(item_kind, name)
    opts = options_of(metric, options)
    return fn(items, opts), metric, opts


def _vectors(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    for it in items[1:]:
        S.same_space(items[0], it)
    return np.array([it["vector"] for it in items], dtype=np.float32)


@_implements("activations/vector", "cosine")
def cosine(items, options):
    x = _vectors(items)
    if options.get("center"):
        x = x - x.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normed = x / np.clip(norms, 1e-12, None)
    return (normed @ normed.T).astype(np.float64)


@_implements("activations/vector", "euclidean")
def euclidean(items, options):
    x = _vectors(items).astype(np.float64)
    sq = (x * x).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    return np.sqrt(np.clip(d2, 0.0, None))


@_implements("activations/vector", "dot")
def dot(items, options):
    x = _vectors(items)
    return (x @ x.T).astype(np.float64)


@_implements("adapter/delta", "cosine")
def delta_cosine(items, options):
    missing = [it.get("id") for it in items if not it.get("vector")]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(items)} deltas carry no vector "
            f"(first: {missing[0]!r}) — `adapter/measure` records the "
            "principal direction only with `vectors: true`")
    modules = {(it.get("basis") or {}).get("module") or
               (it.get("coords") or {}).get("module") for it in items}
    if len(modules) > 1:
        raise ValueError(
            f"these deltas span {len(modules)} modules "
            f"({', '.join(sorted(str(m) for m in modules))}) and each module's "
            "output is its own space — group by module (`by: \"module\"`) "
            "before comparing")
    x = np.array([it["vector"] for it in items], dtype=np.float32)
    normed = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return (normed @ normed.T).astype(np.float64)


def _supports(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    dists = [S.distribution_of(it) for it in items]
    keys: list[Any] = []
    seen: set[Any] = set()

    def key_of(entry: Mapping[str, Any]) -> Any:
        tok = entry.get("token")
        if isinstance(tok, Mapping):
            return tok.get("id") if tok.get("id") is not None else tok.get("text")
        return tok

    masses: list[dict[Any, float]] = []
    for d in dists:
        m: dict[Any, float] = {}
        tracked = [v for t in (d.get("tracked") or {}).values()
                   for v in (t.get("variants") or [t])]
        for entry in list(d.get("top") or []) + tracked:
            k = key_of(entry)
            p = entry.get("p")
            if p is None and entry.get("logp") is not None:
                p = float(np.exp(entry["logp"]))
            if k is None or p is None:
                continue
            m[k] = float(p)
            if k not in seen:
                seen.add(k)
                keys.append(k)
        masses.append(m)
    rows = np.zeros((len(items), len(keys) + 1), dtype=np.float64)
    for i, m in enumerate(masses):
        for j, k in enumerate(keys):
            rows[i, j] = m.get(k, 0.0)
        rows[i, -1] = max(0.0, 1.0 - rows[i, :-1].sum())
    return rows


def _pairwise(rows: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    n = rows.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            out[i, j] = fn(rows[i], rows[j]) if i != j else 0.0
    return out


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float((p[mask] * np.log2(p[mask] / np.clip(q[mask], 1e-12, None))).sum())


@_implements("logits/distribution", "jensen-shannon")
def jensen_shannon(items, options):
    rows = _supports(items)

    def js(p, q):
        m = 0.5 * (p + q)
        return float(np.sqrt(max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))))
    return _pairwise(rows, js)


@_implements("logits/distribution", "hellinger")
def hellinger(items, options):
    rows = _supports(items)
    return _pairwise(rows, lambda p, q: float(np.sqrt(max(0.0, 1.0 - np.sqrt(p * q).sum()))))


@_implements("logits/distribution", "total-variation")
def total_variation(items, options):
    rows = _supports(items)
    return _pairwise(rows, lambda p, q: float(0.5 * np.abs(p - q).sum()))


@_implements("logits/distribution", "kl")
def kl(items, options):
    rows = _supports(items)
    return _pairwise(rows, _kl)


@_implements("records/record", "hamming")
def hamming(items, options):
    coords = [dict(S.coords_of(it)) for it in items]
    axes = sorted({k for c in coords for k in c})
    n = len(items)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = sum(1 for a in axes if coords[i].get(a) != coords[j].get(a))
            out[i, j] = out[j, i] = float(d)
    return out
