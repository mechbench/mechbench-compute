"""The chunk-isomorphism test (task 000407): the map/reduce law as a
property test, run in CI over the catalog and by the api at seal for
chunked run sets.

For a reduce block and a leaf set: random partitions of the leaves,
nested to random depth, must reduce to the flat result — exactly (by
canonical-CBOR digest) for `collect` blocks and exact monoids, within
an envelope for operations declared float-sensitive. `ordered` blocks
are refused from chunking, which the test asserts too.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import reduce as rd


def digest(value: Any) -> str:
    from mechbench_schema import dump_canonical

    return hashlib.sha256(dump_canonical(value)).hexdigest()


def random_partition(leaves: Sequence[Any], rng: random.Random, *,
                     max_depth: int = 3) -> list:
    """A random nested partition of `leaves` preserving leaf order
    within chunks: chunks of chunks to `max_depth`."""
    n = len(leaves)
    if n <= 1 or max_depth == 0:
        return [list(leaves)]
    k = rng.randint(1, min(n, 5))
    cuts = sorted(rng.sample(range(1, n), k - 1)) if k > 1 else []
    bounds = [0, *cuts, n]
    chunks = [list(leaves[bounds[i]:bounds[i + 1]]) for i in range(len(bounds) - 1)]
    if rng.random() < 0.5:
        return [random_partition(c, rng, max_depth=max_depth - 1) for c in chunks]
    return chunks


def _flatten_partition(part) -> list:
    """Nested chunks -> the leaf list in order (a partition never
    reorders leaves)."""
    if part and isinstance(part[0], list) and part and not isinstance(part[0], Mapping):
        out = []
        for p in part:
            out.extend(_flatten_partition(p))
        return out
    return list(part)


def _reduce_nested(monoid: rd.Monoid, part, params) -> Any:
    """Partial over a nested partition: leaves at any depth, merged with
    the canonical tree at each level."""
    if part and isinstance(part[0], list) and not isinstance(part[0], Mapping):
        return rd.merge_tree(monoid, [_reduce_nested(monoid, p, params) for p in part])
    return monoid.partial(part, params)


def _leaf_numbers(x: Any, out: list[float]) -> None:
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, Mapping):
        for k in sorted(x):
            _leaf_numbers(x[k], out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _leaf_numbers(v, out)


def within_envelope(a: Any, b: Any, rel: float) -> bool:
    la: list[float] = []
    lb: list[float] = []
    _leaf_numbers(a, la)
    _leaf_numbers(b, lb)
    if len(la) != len(lb):
        return False
    return all(abs(x - y) <= rel * max(abs(x), abs(y), 1e-12) for x, y in zip(la, lb, strict=True))


def check(block: str, leaves: Sequence[Mapping[str, Any]], params: Mapping[str, Any],
          *, trials: int = 20, seed: int = 0, envelope: float | None = None,
          port: str = "records", inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the law for one block. `port` names the leaf-stream input and
    `inputs` the block's other inputs (unchunked, passed verbatim).
    Returns a report; raises AssertionError with the first violating
    partition on failure."""
    from mechbench_compute.blocks import PURE_BLOCKS

    alg = rd.algebra(block)
    rng = random.Random(seed)
    report: dict[str, Any] = {"block": block, "algebra": alg, "trials": trials,
                              "n_leaves": len(leaves)}
    if alg == "ordered":
        try:
            rd.reduce_chunks(block, [list(leaves)], params, port=port, inputs=inputs)
        except ValueError:
            report["refused"] = True
            return report
        raise AssertionError(f"{block} is declared ordered but reduce_chunks did not refuse it")

    if alg == "collect":
        flat = PURE_BLOCKS[block]({**(inputs or {}), port: list(leaves)}, params)
    else:
        m0 = rd.monoid_for(block, params)
        flat = m0.finalize(m0.partial(list(leaves), params), params)
    ref = digest(flat)
    for t in range(trials):
        part = random_partition(list(leaves), rng)
        if alg == "collect":
            out = rd.reduce_chunks(block, [_flatten_partition(p) for p in part],
                                   params, port=port, inputs=inputs)
        else:
            m = rd.monoid_for(block, params)
            out = m.finalize(_reduce_nested(m, part, params), params)
        if digest(out) == ref:
            continue
        if envelope is not None and within_envelope(out, flat, envelope):
            report.setdefault("envelope_hits", 0)
            report["envelope_hits"] += 1
            continue
        raise AssertionError(
            f"{block}: partition {t} (seed {seed}) reduced to a different result "
            f"than the flat reduce; algebra={alg}")
    report["exact"] = report.get("envelope_hits", 0) == 0
    return report
