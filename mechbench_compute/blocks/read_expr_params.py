from __future__ import annotations

import re

_SEP = re.compile(r"[.\[]")


def read_expr_params(src: str) -> frozenset[str]:
    from mechbench_compute.expr.engine import load_engine

    names: set[str] = set()
    for path in load_engine().check(src)["reads"]:
        head, *rest = _SEP.split(path, 2)
        names.add(rest[0] if head == "params" and rest else head)
    return frozenset(names)
