from __future__ import annotations


def restore_node(state, nid, entry, block, params, fingerprint,
                 on_node_done=None) -> bool:
    from mechbench_compute import bench
    from mechbench_compute import resume as resume_mod

    if not entry:
        return False
    if entry.get("done"):
        fetched = bench.fetch(entry["done"])
        state.results[nid] = (fetched.get("payload", fetched)
                              if isinstance(fetched, dict) else fetched)
        state.node_paths[nid] = entry["done"]
        state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
        if on_node_done is not None:
            on_node_done(nid, state.node_paths[nid], fingerprint)
        return True
    if "held" in entry and state.discard and not state.outputs_of.get(nid):
        state.results[nid] = entry["held"]
        state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
        state.held[nid] = (block, params)
        return True
    return False
