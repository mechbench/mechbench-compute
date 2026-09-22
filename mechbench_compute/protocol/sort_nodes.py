"""The order a graph's nodes run in."""

from __future__ import annotations

from typing import Any


def sort_nodes(nodes: dict[str, Any], edges: list[dict]) -> list[str]:
    """The node ids in topological order (Kahn), refusing a cycle.

    The API validates acyclicity before a job is queued; a runner never
    trusts its inputs to be well-formed.
    """
    indeg = {nid: 0 for nid in nodes}
    for e in edges:
        indeg[e["to"]["node"]] += 1
    order = [nid for nid, d in indeg.items() if d == 0]
    i = 0
    while i < len(order):
        for e in edges:
            if e["from"]["node"] == order[i]:
                t = e["to"]["node"]
                indeg[t] -= 1
                if indeg[t] == 0:
                    order.append(t)
        i += 1
    if len(order) != len(nodes):
        raise ValueError("pipeline graph has a cycle")
    return order
