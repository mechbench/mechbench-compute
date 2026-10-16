from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute.protocol.gather_inputs import gather_inputs
from mechbench_compute.protocol.named_outputs import read_edge_hash
from mechbench_compute.protocol.serialize_params import serialize_params
from mechbench_compute.protocol.sort_edges import sort_edges
from mechbench_compute.registry import REGISTRY


def read_fingerprint(state, block, params, in_edges, inline_hashes) -> str:
    from mechbench_compute import __version__ as core_version
    from mechbench_compute import resume as resume_mod

    serialized = serialize_params(params)
    return resume_mod.node_fingerprint(
        block=REGISTRY.resolve(block).pinned, params=serialized,
        input_hashes=[read_edge_hash(state, e) for e in in_edges] + inline_hashes,
        core_version=core_version, model=str(serialized.get("model", "")))


def read_peer_fingerprint(state, nid, resolver) -> str:
    node = state.nodes[nid]
    block = lexicon.resolve(node["block"])
    params = resolver.resolve_node_params(node, block)
    in_edges = sort_edges(state.edges, nid)
    gathered = gather_inputs(state, nid, node, lexicon.BY_NAME.get(block), in_edges, resolver)
    return read_fingerprint(state, block, params, in_edges, gathered[2])
