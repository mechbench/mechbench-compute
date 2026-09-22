"""What the interpretability operations share: the readouts, resolvers
and shapes more than one of them needs, one to a file.

Every op renders its records one way (`distill.render`): a condition
(`user`, optional `system` and `prefill`) through the model's chat
template, a bare `text`/`prompt` record raw. So an ablation sweep can
read at a decision point inside an assistant turn, exactly where
`logits/read` reads.
"""

from __future__ import annotations

from mechbench_compute.interp.collect_tracked_ids import collect_tracked_ids  # noqa: F401
from mechbench_compute.interp.constants import MAX_VECTOR_FLOATS  # noqa: F401
from mechbench_compute.interp.read_last_logp import read_last_logp  # noqa: F401
from mechbench_compute.interp.read_pair import read_pair  # noqa: F401
from mechbench_compute.interp.render_text import render_text  # noqa: F401
from mechbench_compute.interp.resolve_layers import resolve_layers  # noqa: F401
from mechbench_compute.interp.resolve_target import resolve_target  # noqa: F401
