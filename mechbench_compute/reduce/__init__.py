"""Reduce algebra, under the map/reduce isomorphism law of
DATAFLOW.md.

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

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mechbench_compute.reduce.find_monoid import find_monoid  # noqa: F401
from mechbench_compute.reduce.monoid import Monoid  # noqa: F401

ALGEBRAS = ("collect", "monoid", "ordered")

#: Block ref -> algebra. Pure reduces default to `collect` (they take
#: whole record lists today). Generators (factor-cross, grid, template)
#: are not reduces and are listed for completeness.
REDUCE_ALGEBRA: dict[str, str] = {
    "records/summarize": "monoid",
    "records/union": "collect",
    "records/select": "collect",
    "records/subtract": "collect",
    "records/tabulate": "collect",
    "text/measure": "collect",
    "eval/expect": "collect",
    "geometry/compare": "collect",
}


def algebra(block: str) -> str:
    from mechbench_compute import lexicon

    try:
        block = lexicon.resolve(block, warn=False)
    except KeyError:
        pass
    if block in REDUCE_ALGEBRA:
        return REDUCE_ALGEBRA[block]
    # An operation in its own file says it can be computed in chunks by
    # defining MONOID there (docs/OPS_LAYOUT.md); that is the declaration.
    from mechbench_compute import ops

    return "monoid" if getattr(ops.find(block), "MONOID", None) is not None else "collect"


# --- the monoid interface -------------------------------------------------------


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


MONOIDS: dict[str, Callable[[], Monoid]] = {
}


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
        m = find_monoid(block, params)
        if m is None:
            raise ValueError(f"{block} declares monoid but has no Monoid implementation")
        return m.finalize(merge_tree(m, [m.partial(c, params) for c in chunks]), params)
    from mechbench_compute.blocks import PURE_BLOCKS

    fn = PURE_BLOCKS[block]
    leaves = [r for c in chunks for r in c]
    return fn({**(inputs or {}), port: leaves}, params)


# --- pure-block adapters for the generic monoids ------------------------------------


PURE_REDUCE_BLOCKS = {
}
for _b in PURE_REDUCE_BLOCKS:
    REDUCE_ALGEBRA[_b] = "monoid"
