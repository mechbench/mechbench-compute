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

from typing import Any


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


from mechbench_compute.blocks.expand_cells import expand_cells  # noqa: F401
from mechbench_compute.blocks.expand_grid import expand_grid  # noqa: F401
from mechbench_compute.blocks.read_group_key import read_group_key  # noqa: F401
from mechbench_compute.blocks.read_items import read_items  # noqa: F401
