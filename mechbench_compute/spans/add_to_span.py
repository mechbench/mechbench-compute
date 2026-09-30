from __future__ import annotations

from mechbench_compute.spans.node_span import NodeSpan


def add_to_span(**counts) -> None:
    span = NodeSpan.active
    if span is not None:
        span.add(**counts)
