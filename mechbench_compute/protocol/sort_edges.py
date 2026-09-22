"""The edges into a node, in the one order the platform reads them."""

from __future__ import annotations

from typing import Any


def sort_edges(edges, nid: str) -> list[Any]:
    """The edges into a node, in the one order the platform reads them:
    port, then the edge's declared `index`, then the source node's id.
    Every use of a node's in-edges — its inputs, its lineage, its
    fingerprint — reads this order, so none of them depends on where in
    the graph's edge list an author put a line."""
    into = [e for e in edges if (e.get("to") or {}).get("node") == nid]
    return sorted(into, key=lambda e: (str(e["to"].get("port", "")),
                                       int(e.get("index", 0)),
                                       str((e.get("from") or {}).get("node", ""))))
