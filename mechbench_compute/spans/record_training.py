from __future__ import annotations

from mechbench_compute.spans.node_span import NodeSpan


def record_training(**fields) -> None:
    span = NodeSpan.active
    if span is not None:
        span.training = dict(fields)
