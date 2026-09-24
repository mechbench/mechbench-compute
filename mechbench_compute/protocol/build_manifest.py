from __future__ import annotations

from typing import Any

from mechbench_compute.protocol.sort_edges import sort_edges
from mechbench_compute.protocol.summarize_node import summarize_node
from mechbench_compute.protocol.total_spend import total_spend


def build_manifest(state, resolved: dict) -> dict[str, Any]:
    from mechbench_compute.seeds import hardware_class

    def sanitize(v, at):
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
        "node_hashes": {nid: state.node_hashes[nid] for nid in state.order
                        if nid in state.node_hashes},
        "node_inputs": {nid: [e["from"]["node"]
                              for e in sort_edges(state.edges, nid)]
                        for nid in state.order
                        if nid in state.results and nid not in state.missing},
        **({"keep": state.keep, "nodes_held": sorted(state.held)}
           if state.discard else {}),
        "node_summaries": {nid: summarize_node(state.results[nid],
                                               state.spend_by_node.get(nid))
                           for nid in state.order
                           if nid in state.results and nid not in state.missing},
        **({"nodes_missing": {nid: state.missing[nid] for nid in state.order
                              if nid in state.missing}} if state.missing else {}),
        "resolved": resolved,
        "resources": {"hardware": hardware_class(),
                      **({"spend": total_spend(state.spend_by_node)}
                         if state.spend_by_node else {})},
    }
