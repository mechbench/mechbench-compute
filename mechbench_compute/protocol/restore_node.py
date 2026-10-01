from __future__ import annotations

from mechbench_compute.protocol.named_outputs import (
    find_named_outputs,
    name_output_targets,
    read_edge_output,
)


def restore_node(state, nid, entry, block, params, fingerprint,
                 on_node_done=None) -> bool:
    from mechbench_compute import bench
    from mechbench_compute import resume as resume_mod

    if not entry:
        return False
    named = find_named_outputs(block)
    if entry.get("done"):
        if named and not restore_named_results(state, nid, named):
            return False
        fetched = bench.fetch(entry["done"])
        state.results[nid] = (fetched.get("payload", fetched)
                              if isinstance(fetched, dict) else fetched)
        state.node_paths[nid] = entry["done"]
        state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
        if on_node_done is not None:
            on_node_done(nid, state.node_paths[nid], fingerprint)
        return True
    if "held" in entry and state.discard and not state.outputs_of.get(nid) and not named:
        state.results[nid] = entry["held"]
        state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
        state.held[nid] = (block, params)
        return True
    return False


def restore_named_results(state, nid, named) -> bool:
    from mechbench_compute import bench
    from mechbench_compute import resume as resume_mod

    wanted = sorted({read_edge_output(e) for e in state.edges if e["from"]["node"] == nid}
                    | {name for (source, name) in state.named_outputs_of if source == nid})
    state.named_results[nid] = {}
    for name in (w for w in wanted if w in named):
        path = name_output_targets(state, nid, name)[0]
        try:
            fetched = bench.fetch(path)
        except bench.BenchError:
            print(f"[resume] {nid}: its output {name!r} is not at {path}; recomputing")
            state.named_results.pop(nid, None)
            return False
        value = fetched.get("payload", fetched) if isinstance(fetched, dict) else fetched
        state.named_results[nid][name] = value
        state.named_hashes.setdefault(nid, {})[name] = resume_mod.content_hash(value)
        state.named_paths.setdefault(nid, {})[name] = path
    return True
