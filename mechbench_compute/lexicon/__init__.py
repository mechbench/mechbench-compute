"""The operation lexicon: every canonical op, described for the person
who will use it.

One declaration, three consumers. `check_params` refuses a parameter
that is not declared here; `tests/test_block_params.py` proves the
declaration equals what the code reads, in both directions; and the
documentation site renders these entries as the operation reference.
Because all three read the same source, a documented parameter the
block does not read fails a test, and a parameter the block reads but
nobody documented fails the same test.

Entries are grouped by family in the sibling modules and assembled here.
"""

from __future__ import annotations

from mechbench_compute.lexicon import (
    direction,
    external,
    model,
    records,
    trajectory,
)
from mechbench_compute.lexicon._base import REQUIRED, Op, P, Param
from mechbench_compute.lexicon.common import COMMON

OPS: tuple[Op, ...] = tuple(
    sorted(
        (*model.OPS, *direction.OPS, *trajectory.OPS, *records.OPS, *external.OPS),
        key=lambda op: op.ref,
    )
)

BY_REF: dict[str, Op] = {op.ref: op for op in OPS}

__all__ = ["BY_REF", "COMMON", "OPS", "REQUIRED", "Op", "P", "Param"]
