from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute.protocol.copy_arch import copy_arch
from mechbench_compute.protocol.named_outputs import (
    is_named_edge,
    name_output_targets,
    read_edge_path,
)
from mechbench_compute.protocol.serialize_params import serialize_params
from mechbench_compute.registry import REGISTRY


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
        pin = REGISTRY.resolve(block).pin
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
            inputs=read_cited_inputs(state, in_edges, resolver, node),
            operation=lexicon.canonical_path(block),
            extension=pin,
            params=serialize_params(params),
        )
        state.node_paths[nid] = out["path"]
        for also in names[1:]:
            bench.emit(f"{state.result_base}/{also}", to_emit,
                       inputs=[out["path"]],
                       operation=lexicon.canonical_path(block),
                       extension=pin,
                       params=serialize_params(params))
    if nid in state.named_results:
        store_named_results(state, nid, block, params, in_edges, resolver, node)
    if isinstance(state.results[nid], dict) and state.results[nid].get("spend"):
        state.spend_by_node[nid] = state.results[nid]["spend"]


def read_cited_inputs(state, in_edges, resolver, node) -> list[str]:
    def cite(e):
        source = e["from"]["node"]
        if is_named_edge(state, e):
            return read_edge_path(state, e) or None
        return (state.node_paths.get(source)
                or (f"~hash/sha256:{state.node_hashes[source]}"
                    if source in state.held else None))

    return list(dict.fromkeys([
        *(cited for cited in map(cite, in_edges) if cited is not None),
        *resolver.read_stored_inputs(node)]))


def store_named_results(state, nid, block, params, in_edges, resolver, node) -> None:
    from mechbench_compute import bench
    from mechbench_compute import resume as resume_mod

    pin = REGISTRY.resolve(block).pin
    for name, value in list(state.named_results[nid].items()):
        value = lexicon.canonical_collection(value)
        state.named_results[nid][name] = value
        state.named_hashes.setdefault(nid, {})[name] = resume_mod.content_hash(value)
        declared = state.named_outputs_of.get((nid, name))
        if not state.result_base or (state.discard and not declared):
            continue
        first, *also = name_output_targets(state, nid, name)
        out = bench.emit(first, value,
                         inputs=read_cited_inputs(state, in_edges, resolver, node),
                         operation=lexicon.canonical_path(block), extension=pin,
                         params=serialize_params(params))
        state.named_paths.setdefault(nid, {})[name] = out["path"]
        for target in also:
            bench.emit(target, value, inputs=[out["path"]],
                       operation=lexicon.canonical_path(block), extension=pin,
                       params=serialize_params(params))
