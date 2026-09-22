"""What a node produced, hashed and put where the run keeps it.

Hashing comes BEFORE emitting. The hash canonical-encodes the result,
so a result carrying a live object fails here, locally and by name,
rather than reaching the emit path and failing as a transport fault. A
collection is stored with its items in key order: the same items in any
order are the same bytes.

Every emitted node becomes a bench object under the job's result
namespace, with lineage inputs = its upstream nodes' paths and
operation = the block ref, so the protocol graph and the lineage graph
are the same graph by construction.
"""

from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute.protocol.copy_arch import copy_arch
from mechbench_compute.protocol.serialize_params import serialize_params


def store_result(state, nid, node, block, params, in_edges, inputs, resolver,
                 fingerprint, on_node_kept=None) -> None:
    """Canonicalize one node's result, hash it, and either hold it on
    this device or emit it under the run's result path."""
    from mechbench_compute import bench, dataflow
    from mechbench_compute import resume as resume_mod
    from mechbench_compute import tensors as tensors_mod

    state.results[nid] = lexicon.canonical_collection(state.results[nid])
    # A result about a model's layers stays about them through every
    # records op: the landmarks on any input's header ride onto an
    # output that has none.
    state.results[nid] = copy_arch(inputs, state.results[nid])
    state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
    if state.result_base and state.discard and not state.outputs_of.get(nid):
        # Held, not emitted: the consumers read it from memory, a resume
        # from the spool, and the API never sees the bytes. Its identity
        # is its content hash, which the manifest records and its
        # consumers' lineage cites.
        state.held[nid] = (block, params)
        if on_node_kept is not None:
            on_node_kept(nid, fingerprint, state.results[nid])
    elif state.result_base:
        names = state.outputs_of.get(nid, [])
        if state.declared_outputs is None:
            target = f"{state.result_base}/{nid}"
        elif names:
            target = f"{state.result_base}/{names[0]}"
        else:
            target = f"{state.result_base}/{dataflow.INTERMEDIATES}/{nid}"
        to_emit = state.results[nid]
        if tensors_mod.is_tensor(to_emit):
            # The rows go up first as raw shards under the result's
            # label (retry-as-resume: a shard already there with the
            # same hash is not sent again), then the header is emitted
            # as the object itself. The in-memory result keeps its local
            # shard dir: a consumer in this job reads the rows from
            # there.
            to_emit = tensors_mod.upload(
                to_emit, target,
                lambda lab, path: bench.put_file(lab, path, kind="tensor_shard"),
                have=bench.list_prefix_hashes(target))
        out = bench.emit(
            target,
            to_emit,
            # Lineage names the inputs that EXIST. A node run under
            # `on_missing` has an upstream that produced nothing and so
            # stored nothing: there is no path to cite, and the absence
            # is recorded under `nodes_missing` on the manifest instead.
            # An upstream HELD rather than stored (discard mode) is
            # cited by its content hash, which is a path form of its own.
            inputs=list(dict.fromkeys([
                *(cited for cited in (
                    state.node_paths.get(e["from"]["node"])
                    or (f"~hash/sha256:{state.node_hashes[e['from']['node']]}"
                        if e["from"]["node"] in state.held else None)
                    for e in in_edges) if cited is not None),
                # …and the stored objects it read by reference.
                *resolver.read_stored_inputs(node)])),
            # Provenance records the stored identity.
            operation=lexicon.canonical_path(block),
            params=serialize_params(params),
        )
        state.node_paths[nid] = out["path"]
        # A node declared as several outputs is stored under each name;
        # the first is the one its consumers' lineage cites.
        for also in names[1:]:
            bench.emit(f"{state.result_base}/{also}", to_emit,
                       inputs=[out["path"]],
                       operation=lexicon.canonical_path(block),
                       params=serialize_params(params))
    if isinstance(state.results[nid], dict) and state.results[nid].get("spend"):
        state.spend_by_node[nid] = state.results[nid]["spend"]
