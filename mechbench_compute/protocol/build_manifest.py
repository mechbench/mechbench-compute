"""The record a run leaves: what it produced, where each node's object
went, what each node hashed to, what did not run and why, what
resolved, and where it ran.

A reader of this must never have to infer an absence from a shorter
list, which is why the nodes that produced nothing are named as well as
the ones that did.
"""

from __future__ import annotations

from typing import Any

from mechbench_compute.protocol.sort_edges import sort_edges
from mechbench_compute.protocol.summarize_node import summarize_node
from mechbench_compute.protocol.total_spend import total_spend


def build_manifest(state, resolved: dict) -> dict[str, Any]:
    """The `run/result` payload for a walk that reached the end."""
    from mechbench_compute.seeds import hardware_class

    def sanitize(v, at):
        # Manifests reference binary, never embed it: bytes are replaced
        # by a stub pointing at the node object that holds the real
        # payload.
        if isinstance(v, bytes):
            return {"$binary": {"bytes": len(v), "stored_at": at}}
        if isinstance(v, dict):
            return {k: sanitize(x, at) for k, x in v.items()}
        if isinstance(v, list):
            return [sanitize(x, at) for x in v]
        return v

    if state.declared_outputs is None:
        terminals = [nid for nid in state.nodes
                     if not any(e["from"]["node"] == nid for e in state.edges)
                     and nid not in state.missing]
        kept = {nid: nid for nid in terminals}
    else:
        # By declared name. An output whose node produced nothing is
        # absent here and named under `nodes_missing`.
        kept = {o["name"]: o["from"]["node"] for o in state.declared_outputs
                if o["from"]["node"] in state.results
                and o["from"]["node"] not in state.missing}
    return {
        "kind": "run/result",
        "outputs": {name: sanitize(state.results[nid],
                                   state.node_paths.get(nid, ""))
                     for name, nid in kept.items()},
        **({"output_nodes": dict(kept)}
           if state.declared_outputs is not None else {}),
        "nodes_executed": [nid for nid in state.order
                           if nid not in state.missing],
        "node_paths": state.node_paths,
        # Every node's content hash and its upstream nodes: what a
        # result's lineage verifies against when an intermediate was held
        # rather than stored — and a record worth having either way.
        "node_hashes": {nid: state.node_hashes[nid] for nid in state.order
                        if nid in state.node_hashes},
        "node_inputs": {nid: [e["from"]["node"]
                              for e in sort_edges(state.edges, nid)]
                        for nid in state.order
                        if nid in state.results and nid not in state.missing},
        **({"keep": state.keep, "nodes_held": sorted(state.held)}
           if state.discard else {}),
        # What each node produced, small enough to read beside the node
        # in the composer without fetching its object: the kind, and how
        # many items or rows.
        "node_summaries": {nid: summarize_node(state.results[nid],
                                               state.spend_by_node.get(nid))
                           for nid in state.order
                           if nid in state.results and nid not in state.missing},
        **({"nodes_missing": {nid: state.missing[nid] for nid in state.order
                              if nid in state.missing}} if state.missing else {}),
        "resolved": resolved,
        # Where this ran. Recorded, never fingerprinted: bit-identity is
        # promised within a hardware class.
        "resources": {"hardware": hardware_class(),
                      **({"spend": total_spend(state.spend_by_node)}
                         if state.spend_by_node else {})},
    }
