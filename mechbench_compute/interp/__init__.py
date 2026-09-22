"""What the interpretability operations share: the readouts, resolvers
and shapes more than one of them needs.

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
from mechbench_compute.interp.load_kinds import load_kinds  # noqa: F401
from mechbench_compute.interp.read_last_logp import read_last_logp  # noqa: F401
from mechbench_compute.interp.read_pair import read_pair  # noqa: F401
from mechbench_compute.interp.render_text import render_text  # noqa: F401
from mechbench_compute.interp.resolve_layers import resolve_layers  # noqa: F401
from mechbench_compute.interp.resolve_target import resolve_target  # noqa: F401
from mechbench_compute.interp.encode_target_token import encode_target_token  # noqa: F401
from mechbench_compute.interp.collect_tracked_ids import collect_tracked_ids  # noqa: F401
from mechbench_compute.interp.constants import MAX_VECTOR_FLOATS  # noqa: F401
from mechbench_compute.interp.read_record_coords import read_record_coords  # noqa: F401
from mechbench_compute.interp.report_own_top1 import report_own_top1  # noqa: F401


