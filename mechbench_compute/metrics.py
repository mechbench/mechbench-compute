"""The metrics bound to kinds (`Kind.metrics` in the lexicon), and the
one function that applies one to a collection.

A metric is declared on the kind it compares — `cosine` on
`activations/vector`, `jensen-shannon` on `logits/distribution`,
`hamming` on `records/record` — and a subtype inherits its ancestor's.
`geometry/compare` asks this module for the pairwise matrix of a
collection under a named metric; `geometry/span` turns that matrix into
a tree. So "mst over anything comparable" is one op over any kind that
declares how its items compare.

Every implementation here takes the whole item list and returns the
`[n, n]` matrix, because the vector metrics are a matrix product and a
pairwise loop would be the wrong shape for them. The result is always a
float64 array of Python-JSON-safe numbers; the vector metrics compute in
float32, which is what the geometry readouts compute in, so both paths
produce the same similarity bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon._base import Metric

#: (kind name, metric name) -> the function producing the matrix.
IMPLEMENTATIONS: dict[tuple[str, str], Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], np.ndarray]] = {}


def _implements(kind: str, name: str):
    def deco(fn):
        IMPLEMENTATIONS[(kind, name)] = fn
        return fn
    return deco


def declared(item_kind: str) -> dict[str, Metric]:
    """The metrics an item kind offers, its ancestors' included; the
    nearest declaration wins by name."""
    out: dict[str, Metric] = {}
    for name in reversed(K.ancestry(item_kind)):
        for m in K.BY_KIND[name].metrics:
            out[m.name] = m
    return out


def default_metric(item_kind: str) -> str | None:
    """The metric a kind compares by when the protocol names none: the
    first its nearest declaring ancestor lists."""
    for name in K.ancestry(item_kind):
        if K.BY_KIND[name].metrics:
            return K.BY_KIND[name].metrics[0].name
    return None


def resolve(item_kind: str, name: str | None) -> tuple[str, Metric, Callable]:
    """(the declaring kind, the metric, its implementation) for `name`
    on `item_kind`, or the kind's default. Refuses a metric the kind
    does not declare, naming the ones it does."""
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
    """The metric's options with defaults filled in; an option the
    metric does not declare is refused by name."""
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
    """The pairwise matrix of `items` under the named metric: `(matrix,
    metric, options as applied)`."""
    _, metric, fn = resolve(item_kind, name)
    opts = options_of(metric, options)
    return fn(items, opts), metric, opts


# --- activations/vector ---------------------------------------------------------------


def _vectors(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """The items' vectors as one float32 matrix, refusing two that do
    not share a space — with both spaces in the message."""
    for it in items[1:]:
        S.same_space(items[0], it)
    return np.array([it["vector"] for it in items], dtype=np.float32)


@_implements("activations/vector", "cosine")
def cosine(items, options):
    x = _vectors(items)
    if options.get("center"):
        # Transformer representations occupy a narrow cone around one
        # dominant direction; subtracting the corpus mean removes it, so
        # the cosine measures how two items differ rather than the cone.
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


# --- adapter/delta ---------------------------------------------------------------------


@_implements("adapter/delta", "cosine")
def delta_cosine(items, options):
    """Two adapters' writes at one module, compared through the
    principal direction of each.

    Not `activations/vector`'s cosine: these live in a module's OUTPUT
    space, not a layer's residual, so there is no `space` to agree on —
    what must agree is the module, and a comparison across modules is
    refused here rather than producing a number for two different
    spaces. A sign-free reading is deliberate: `u` and `−u` are the same
    principal axis, so the magnitude is what carries the alignment.
    """
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


# --- logits/distribution --------------------------------------------------------------


def _supports(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Each distribution as a row over the union of the tokens the items
    carry (`top` and `tracked`, by token id, else by text), with the mass
    none of them names as one last bucket — a read keeps its top tokens,
    not the vocabulary, so this is the comparison the data allows."""
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
        for entry in list(d.get("top") or []) + list((d.get("tracked") or {}).values()):
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
    """The Jensen–Shannon distance: the square root of the divergence
    in bits, in [0, 1]."""
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
    """KL(p ‖ q) in bits, row p against column q. Not symmetric."""
    rows = _supports(items)
    return _pairwise(rows, _kl)


# --- records/record --------------------------------------------------------------------


@_implements("records/record", "hamming")
def hamming(items, options):
    """How many coordinate axes two records differ on; an axis one of
    them lacks counts as a difference."""
    coords = [dict(S.coords_of(it)) for it in items]
    axes = sorted({k for c in coords for k in c})
    n = len(items)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = sum(1 for a in axes if coords[i].get(a) != coords[j].get(a))
            out[i, j] = out[j, i] = float(d)
    return out
