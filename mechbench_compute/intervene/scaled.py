from __future__ import annotations

from collections.abc import Sequence

from mechbench_compute.intervene.spec import Spec


def scaled(specs: Sequence[Spec], factor: float) -> list[Spec]:
    """The specs at one sweep factor: each strength multiplied. Factor 1
    is the specs themselves."""
    if factor == 1.0:
        return list(specs)
    out = []
    for spec in specs:
        s2 = Spec.__new__(Spec)
        s2.__dict__.update(spec.__dict__)
        s2.strength = spec.strength * factor
        out.append(s2)
    return out
