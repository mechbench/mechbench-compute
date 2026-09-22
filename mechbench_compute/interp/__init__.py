"""Interpretability primitives as protocol operations (the
mechbench-experiments port).

The original step-XX scripts each hand-rolled a loop around the same
three moves: run the model with an intervention, read something out of
the residual stream, compare. The intervention layer (interventions.py)
already made those moves declarative; this module makes them PROTOCOL
BLOCKS, so the experiments become graphs anyone can run, re-run, and
diff on the platform:

- ``ablate_layers``     — steps 02/04/34/35 and the legacy flat kind:
                          per-layer (or per-sublayer) Δ log p sweeps.
- ``residual_vectors``  — steps 01/08/10/11/12's shared substrate:
                          residual-stream vectors at (layer, position)
                          per condition, as data other blocks consume.
- ``residual_divergence`` — the matched-pair mechanism (000050/052):
                          run a pair of prompts, cosine-compare the
                          residual streams per (layer, position).
- ``vector_similarity`` — steps 10/11/28's readout (pure, no model):
                          cosine matrix + separation metrics over
                          labeled vectors. Lives in blocks.PURE_BLOCKS.

Every op renders its records one way (`distill.render`): a condition
(`user`, optional `system` and `prefill`) through the model's chat
template, a bare `text`/`prompt` record raw. So an ablation sweep can
read at a decision point inside an assistant turn, exactly where
`logits/read` reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from mechbench_compute import points as P
from mechbench_compute import positions as POS
from mechbench_compute import shapes as S
from mechbench_compute.distill import encode, render
from mechbench_compute.interventions import Ablate, Capture
from mechbench_compute.interp.k import _K  # noqa: F401
from mechbench_compute.interp.last_logp import _last_logp  # noqa: F401
from mechbench_compute.interp.pair import _pair  # noqa: F401
from mechbench_compute.interp.render_text import _render_text  # noqa: F401
from mechbench_compute.interp.resolve_layers import _resolve_layers  # noqa: F401
from mechbench_compute.interp.target_of import _target_of  # noqa: F401
from mechbench_compute.interp.target_token_id import _target_token_id  # noqa: F401
from mechbench_compute.interp.tracked_ids import _tracked_ids  # noqa: F401
from mechbench_compute.interp.constants import MAX_VECTOR_FLOATS  # noqa: F401
from mechbench_compute.interp.coords_of import _coords_of  # noqa: F401
from mechbench_compute.interp.own_top1_if_different import _own_top1_if_different  # noqa: F401


