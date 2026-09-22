"""Stdlib building blocks: the pure blocks, plus the block registry the
pipeline executor resolves op refs against.

An operation is named once, in the lexicon, and that name is used
everywhere — here, in a stored graph, on the docs page.

Design rules these implement:

- **`records/cross`**: factors -> records with coordinates —
  the fully crossed design of experimental methodology. A factor
  enumerates its levels or samples them from a seeded generator.
  Records carry {id, coords, values}: `coords` are the level KEYS
  (structured, for `records/summarize` and `records/subtract` — never
  parsed from the id), `values` the substitution payloads (which carry
  their own whitespace; there is no hidden joining logic). "Axis" is
  reserved for the charting surface, deliberately.
- **Range-splitting invariance**: sampled axes derive one rng per
  instance from (seed, index), so generate(seed, 0, 1000) equals
  generate(seed, 0, 100) + generate(seed, 100, 900). Incremental
  dataset growth is the same node run over a later range.
- **`records/fill`**: records x named string templates -> the same
  records with instantiated string fields. Source-agnostic: records may
  come from `records/cross` or from any record stream (dataset rows,
  later).
  No block in this module knows what a "prompt" is.
"""

from __future__ import annotations

import math

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any


# --- registry ---------------------------------------------------------------

# Op ref -> callable. Pure blocks take (inputs, params); model blocks
# are registered by the executor host (the runner), which owns model
# lifecycle.


def arch_header(arch: Any) -> dict[str, Any]:
    """The model's depth landmarks, as a result header carries them: how
    many layers, which attend globally, and where fresh keys and values
    stop. A figure that carries these draws them on its depth axis, so a
    layer is the same place in every figure (VISUALIZATION.md).

    A model family with no hybrid attention or key/value sharing has no
    such landmarks, and says so by leaving them out rather than
    inventing an empty list."""
    out: dict[str, Any] = {"n_layers": int(arch.n_layers)}
    globals_ = getattr(arch, "global_layers", None)
    if globals_ is not None:
        out["global_layers"] = [int(i) for i in globals_]
    first_shared = getattr(arch, "first_kv_shared_layer", None)
    if first_shared is not None:
        out["first_kv_shared_layer"] = int(first_shared)
    return out


PURE_BLOCKS: dict[str, Callable[..., Any]] = {
}


# Trajectory readouts: pure numpy over trajectory records.
from mechbench_compute.trajectory import PURE as _TRAJECTORY_PURE

PURE_BLOCKS.update(_TRAJECTORY_PURE)


# Directions as first-class objects: pure producers and arithmetic live
# in `directions`; the vocabulary projection is a model block in the
# executor.
from mechbench_compute.directions import (
    PURE_DIRECTION_BLOCKS as _DIRECTION_BLOCKS,
)

PURE_BLOCKS.update(_DIRECTION_BLOCKS)

# Exact generic monoid reduces: sum, top-k, histogram.
from mechbench_compute.reduce import PURE_REDUCE_BLOCKS as _REDUCE_BLOCKS

PURE_BLOCKS.update(_REDUCE_BLOCKS)

# Tool handlers are ordinary blocks: what a model may call is what the
# platform can already do.
from mechbench_compute.tools import PURE_TOOL_BLOCKS as _TOOL_BLOCKS

PURE_BLOCKS.update(_TOOL_BLOCKS)

from mechbench_compute.blocks.expand_cells import expand_cells  # noqa: F401
from mechbench_compute.blocks.expand_grid import expand_grid  # noqa: F401
from mechbench_compute.blocks.read_group_key import read_group_key  # noqa: F401
from mechbench_compute.blocks.read_interval import read_interval  # noqa: F401
from mechbench_compute.blocks.read_items import read_items  # noqa: F401
from mechbench_compute.blocks.build_collection import build_collection  # noqa: F401
from mechbench_compute.blocks.pure_blocks import _PureBlocks  # noqa: F401

# An operation that has its own file (docs/OPS_LAYOUT.md) and runs with
# no executor is callable here by name, as the ones above are; the
# lookup happens when asked for, never at import.
PURE_BLOCKS = _PureBlocks(PURE_BLOCKS)


