from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mechbench_compute.reduce.find_monoid import find_monoid
from mechbench_compute.reduce.monoid import Monoid

ALGEBRAS = ("collect", "monoid", "ordered")

REDUCE_ALGEBRA: dict[str, str] = {
    "records/union": "collect",
    "records/derive": "collect",
    "records/filter": "collect",
    "records/join": "collect",
    "records/group": "ordered",
    "records/sort": "ordered",
    "records/tabulate": "collect",
    "text/measure": "collect",
    "eval/expect": "collect",
    "geometry/compare": "collect",
}


def algebra(block: str) -> str:
    from mechbench_compute.registry import REGISTRY

    resolved = REGISTRY.find(block)
    name = resolved.name if resolved is not None else block
    if name in REDUCE_ALGEBRA:
        return REDUCE_ALGEBRA[name]
    return "monoid" if getattr(getattr(resolved, "module", None), "MONOID", None) is not None else "collect"


def merge_tree(monoid: Monoid, partials: Sequence[Any]) -> Any:
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


MONOIDS: dict[str, Callable[[], Monoid]] = {
}


def reduce_chunks(block: str, chunks: Sequence[Sequence[Mapping[str, Any]]],
                  params: Mapping[str, Any], *, port: str = "records",
                  inputs: Mapping[str, Any] | None = None) -> Any:
    alg = algebra(block)
    if alg == "ordered":
        raise ValueError(f"{block} is an ordered reduce: it cannot be chunked")
    if alg == "monoid":
        m = find_monoid(block, params)
        if m is None:
            raise ValueError(f"{block} declares monoid but has no Monoid implementation")
        return m.finalize(merge_tree(m, [m.partial(c, params) for c in chunks]), params)
    from mechbench_compute import ops

    leaves = [r for c in chunks for r in c]
    return ops.run_standalone(block, {**(inputs or {}), port: leaves}, params)
