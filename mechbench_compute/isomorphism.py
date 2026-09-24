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
    if part and isinstance(part[0], list) and part and not isinstance(part[0], Mapping):
        out = []
        for p in part:
            out.extend(_flatten_partition(p))
        return out
    return list(part)


def _reduce_nested(monoid: rd.Monoid, part, params) -> Any:
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
    from mechbench_compute import ops

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
        flat = ops.run_standalone(block, {**(inputs or {}), port: list(leaves)}, params)
    else:
        m0 = rd.find_monoid(block, params)
        flat = m0.finalize(m0.partial(list(leaves), params), params)
    ref = digest(flat)
    for t in range(trials):
        part = random_partition(list(leaves), rng)
        if alg == "collect":
            out = rd.reduce_chunks(block, [_flatten_partition(p) for p in part],
                                   params, port=port, inputs=inputs)
        else:
            m = rd.find_monoid(block, params)
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
