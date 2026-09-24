from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mechbench_compute.weights.constants import MODEL_SCOPE
from mechbench_compute.weights.compile_point_pattern import compile_point_pattern


def select_points(names: Iterable[str], points: Any) -> list[str]:
    have = list(names)
    if points in (None, "all"):
        return have
    if isinstance(points, str):
        points = [points]
    wanted: list[str] = []
    for p in points:
        raw = str(p)
        bare = raw.removeprefix(MODEL_SCOPE)
        matcher = compile_point_pattern(bare)
        hit = [n for n in have if matcher.match(n)]
        if not hit:
            raise ValueError(
                f"no parameter matches {raw!r}. A point names the module "
                f"tree — `layers.12.self_attn.q_proj`, `embed_tokens`, "
                f"`layers.*.mlp.down_proj` — and this model carries "
                f"{len(have)} parameters, e.g. {', '.join(have[:3])}.")
        wanted += [n for n in hit if n not in wanted]
    return [n for n in have if n in set(wanted)]
