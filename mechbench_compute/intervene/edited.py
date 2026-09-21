from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import Any


@contextlib.contextmanager
def edited(model, weight_items: Sequence[Mapping[str, Any]], factor: float):
    """The model's weights edited by `weight_items` at `factor` for the
    block's duration, and put back whatever happens inside — a run that
    left a model edited would poison every later node in the job."""
    if not weight_items or factor == 0.0:
        yield
        return
    from mechbench_compute import weights as weights_mod

    handle = weights_mod.edit_parameters(model.lm, weight_items, factor)
    try:
        yield
    finally:
        weights_mod.restore_parameters(model.lm, handle)
