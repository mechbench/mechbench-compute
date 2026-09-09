"""Reduce algebra (task 000406, epic 000364; the map/reduce isomorphism
law of DATAFLOW.md / task 000405).

A hierarchy of map/reduce must equal the flat iteration over all
leaves. Every reduce block therefore declares its algebra:

  collect   the reduce sees every leaf; chunking is invisible by
            construction (chunk outputs are concatenations of leaves).
  monoid    an identity and an associative, commutative merge over
            PARTIALS, so chunks may reduce partially and merges may
            nest to any depth. The monoids here are EXACT: their
            partials are multisets or exactly-summable values, so
            flat == chunked bit for bit under any partition. (Bounded
            approximate sketches would be a separate declaration with
            envelope-bounded equality; none exist yet.)
  ordered   a sequential fold with non-associative steps; legal only
            over canonical leaf order and never chunked.

Floating point: sums are `math.fsum` (correctly rounded, hence
order-independent), and `merge_tree` merges partials in a balanced
binary tree over the leaf-key range, so association order is a
function of N alone even for merges that are merely associative.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

ALGEBRAS = ("collect", "monoid", "ordered")

#: Block ref -> algebra. Pure reduces default to `collect` (they take
#: whole record lists today). Generators (factor-cross, grid, template)
#: are not reduces and are listed for completeness.
REDUCE_ALGEBRA: dict[str, str] = {
    "~canonical/ops/group-stats/1": "monoid",
    "~canonical/ops/union/1": "collect",
    "~canonical/ops/select/1": "collect",
    "~canonical/ops/paired-delta/1": "collect",
    "~canonical/ops/table/from-records/1": "collect",
    "~canonical/ops/text/stats/1": "collect",
    "~canonical/ops/eval/expectation/1": "collect",
    "~canonical/ops/vectors/similarity/1": "collect",
}


def algebra(block: str) -> str:
    return REDUCE_ALGEBRA.get(block, "collect")


# --- the monoid interface -------------------------------------------------------


class Monoid:
    """`partial(records, params)` → P; `identity()` → P; `merge(P, P)` → P
    (associative, commutative); `finalize(P, params)` → the block's
    output. `partial(all) == merge over any partition of all` exactly."""

    def identity(self) -> Any:
        raise NotImplementedError

    def partial(self, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def merge(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def finalize(self, p: Any, params: Mapping[str, Any]) -> Any:
        raise NotImplementedError


def merge_tree(monoid: Monoid, partials: Sequence[Any]) -> Any:
    """Merge partials in a balanced binary tree over their (leaf-key)
    order: the association order depends only on len(partials)."""
    items = list(partials)
    if not items:
        return monoid.identity()
    while len(items) > 1:
        nxt = []
        for i in range(0, len(items) - 1, 2):
            nxt.append(monoid.merge(items[i], items[i + 1]))
        if len(items) % 2:
            nxt.append(items[-1])
        items = nxt
    return items[0]


# --- exact building blocks -------------------------------------------------------


class FloatSum(Monoid):
    """Exact sum of floats: the partial keeps the values as a multiset
    (a sorted tuple); finalize uses `math.fsum`. Order-independent."""

    def identity(self):
        return ()

    def partial(self, records, params):
        f = params["value"]
        return tuple(sorted(float(r[f]) for r in records))

    def merge(self, a, b):
        return tuple(sorted(a + b))

    def finalize(self, p, params):
        return {"n": len(p), "sum": math.fsum(p)}


class TopK(Monoid):
    """Exact top-k by a value field: merge = top-k of the union."""

    def identity(self):
        return ()

    def partial(self, records, params):
        k = int(params.get("k", 10))
        f = params["value"]
        rows = sorted(records, key=lambda r: (-float(r[f]), str(r.get("id"))))
        return tuple(dict(r) for r in rows[:k])

    def merge(self, a, b):
        return tuple(sorted(a + b, key=lambda r: (-float(r[self._f]), str(r.get("id"))))[: self._k])

    def finalize(self, p, params):
        return {"kind": "record_set", "records": list(p)}

    def bind(self, params):
        self._f = params["value"]
        self._k = int(params.get("k", 10))
        return self


class Histogram(Monoid):
    """Fixed-bin histogram: counts add. Exact."""

    def identity(self):
        return {}

    def partial(self, records, params):
        f = params["value"]
        lo, hi, n = float(params["lo"]), float(params["hi"]), int(params["bins"])
        counts: dict[int, int] = {}
        for r in records:
            v = float(r[f])
            b = n if v >= hi else (-1 if v < lo else int((v - lo) / (hi - lo) * n))
            counts[b] = counts.get(b, 0) + 1
        return counts

    def merge(self, a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0) + v
        return out

    def finalize(self, p, params):
        n = int(params["bins"])
        return {"kind": "histogram", "bins": [p.get(i, 0) for i in range(n)],
                "below": p.get(-1, 0), "above": p.get(n, 0)}


class GroupStats(Monoid):
    """The exact monoid form of `group-stats`: per group, the multiset
    of values (sorted); finalize reproduces the block's rows with
    `fsum` means. Bit-identical to the flat block by construction."""

    def identity(self):
        return {}

    def partial(self, records, params):
        by = params.get("by") or []
        f = params["value"]
        groups: dict[tuple, list[float]] = {}
        for r in records:
            key = tuple(r.get("coords", {}).get(k) for k in by)
            groups.setdefault(key, []).append(float(r[f]))
        return {k: tuple(sorted(v)) for k, v in groups.items()}

    def merge(self, a, b):
        out = {k: v for k, v in a.items()}
        for k, v in b.items():
            out[k] = tuple(sorted(out.get(k, ()) + v))
        return out

    def finalize(self, p, params):
        from statistics import median

        by = params.get("by") or []
        value_field = params["value"]
        rows = []
        for key in sorted(p, key=lambda k: tuple(str(x) for x in k)):
            vals = list(p[key])
            row = {k: key[i] for i, k in enumerate(by)}
            row.update({
                "n": len(vals),
                "median": round(median(vals), 4),
                "mean": round(math.fsum(vals) / len(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "share_negative": round(sum(v < 0 for v in vals) / len(vals), 3),
            })
            rows.append(row)
        columns = [{"name": k, "dtype": "string"} for k in by] + [
            {"name": n, "dtype": "number"}
            for n in ("n", "median", "mean", "min", "max", "share_negative")
        ]
        return {"kind": "metric_table",
                "name": params.get("name", f"{value_field}-stats"),
                "description": params.get("description", ""),
                "row_axis": "condition", "columns": columns, "rows": rows}


MONOIDS: dict[str, Callable[[], Monoid]] = {
    "~canonical/ops/group-stats/1": GroupStats,
    "~canonical/ops/reduce/sum/1": FloatSum,
    "~canonical/ops/reduce/top-k/1": TopK,
    "~canonical/ops/reduce/histogram/1": Histogram,
}


def monoid_for(block: str, params: Mapping[str, Any] | None = None) -> Monoid | None:
    cls = MONOIDS.get(block)
    if cls is None:
        return None
    m = cls()
    if hasattr(m, "bind"):
        m.bind(params or {})
    return m


def reduce_chunks(block: str, chunks: Sequence[Sequence[Mapping[str, Any]]],
                  params: Mapping[str, Any], *, port: str = "records",
                  inputs: Mapping[str, Any] | None = None) -> Any:
    """Reduce a partitioned leaf set: partial per chunk, canonical
    merge tree, finalize. Refuses `ordered` blocks and, for `collect`
    blocks, re-collects the leaves in chunk order and runs the flat
    block (chunking invisible). `port` names the leaf-stream input and
    `inputs` carries the block's other, unchunked inputs (an
    expectations table, a second record stream) verbatim."""
    alg = algebra(block)
    if alg == "ordered":
        raise ValueError(f"{block} is an ordered reduce: it cannot be chunked")
    if alg == "monoid":
        m = monoid_for(block, params)
        if m is None:
            raise ValueError(f"{block} declares monoid but has no Monoid implementation")
        return m.finalize(merge_tree(m, [m.partial(c, params) for c in chunks]), params)
    from mechbench_compute.blocks import PURE_BLOCKS

    fn = PURE_BLOCKS[block]
    leaves = [r for c in chunks for r in c]
    return fn({**(inputs or {}), port: leaves}, params)


# --- pure-block adapters for the generic monoids ------------------------------------


def _block_of(name: str):
    def fn(inputs, params):
        from mechbench_compute.blocks import _records

        m = monoid_for(name, params)
        raw = inputs.get("records") if isinstance(inputs, Mapping) else inputs
        recs = _records(raw if raw is not None else params.get("records"))
        return m.finalize(m.partial(recs, params), params)
    return fn


PURE_REDUCE_BLOCKS = {
    "~canonical/ops/reduce/sum/1": _block_of("~canonical/ops/reduce/sum/1"),
    "~canonical/ops/reduce/top-k/1": _block_of("~canonical/ops/reduce/top-k/1"),
    "~canonical/ops/reduce/histogram/1": _block_of("~canonical/ops/reduce/histogram/1"),
}
for _b in PURE_REDUCE_BLOCKS:
    REDUCE_ALGEBRA[_b] = "monoid"
