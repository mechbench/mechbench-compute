from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute.protocol.copy_arch import copy_arch
from mechbench_compute.protocol.serialize_params import serialize_params


def store_result(state, nid, node, block, params, in_edges, inputs, resolver,
                 fingerprint, on_node_kept=None) -> None:
    from mechbench_compute import bench, dataflow
    from mechbench_compute import resume as resume_mod
    from mechbench_compute import tensors as tensors_mod

    state.results[nid] = lexicon.canonical_collection(state.results[nid])
    state.results[nid] = copy_arch(inputs, state.results[nid])
    state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
    if state.result_base and state.discard and not state.outputs_of.get(nid):
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
            to_emit = tensors_mod.upload(
                to_emit, target,
                lambda lab, path: bench.put_file(lab, path, kind="tensor_shard"),
                have=bench.list_prefix_hashes(target))
        out = bench.emit(
            target,
            to_emit,
            inputs=list(dict.fromkeys([
                *(cited for cited in (
                    state.node_paths.get(e["from"]["node"])
                    or (f"~hash/sha256:{state.node_hashes[e['from']['node']]}"
                        if e["from"]["node"] in state.held else None)
                    for e in in_edges) if cited is not None),
                *resolver.read_stored_inputs(node)])),
            operation=lexicon.canonical_path(block),
            params=serialize_params(params),
        )
        state.node_paths[nid] = out["path"]
        for also in names[1:]:
            bench.emit(f"{state.result_base}/{also}", to_emit,
                       inputs=[out["path"]],
                       operation=lexicon.canonical_path(block),
                       params=serialize_params(params))
    if isinstance(state.results[nid], dict) and state.results[nid].get("spend"):
        state.spend_by_node[nid] = state.results[nid]["spend"]
