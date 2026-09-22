"""A node a previous attempt already finished, put back in place."""

from __future__ import annotations


def restore_node(state, nid, entry, block, params, fingerprint,
                 on_node_done=None) -> bool:
    """True when this node needs no run at all: its whole result is
    already accounted for under this exact fingerprint, and the walk
    goes on to the next node. False when there is work left, which
    includes the partial work an entry's items or checkpoint offer.
    """
    from mechbench_compute import bench
    from mechbench_compute import resume as resume_mod

    if not entry:
        return False
    if entry.get("done"):
        # Its object already exists on the bench under this exact
        # fingerprint, emitted by the earlier attempt. Fetch it as the
        # result.
        fetched = bench.fetch(entry["done"])
        state.results[nid] = (fetched.get("payload", fetched)
                              if isinstance(fetched, dict) else fetched)
        state.node_paths[nid] = entry["done"]
        state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
        if on_node_done is not None:
            on_node_done(nid, state.node_paths[nid], fingerprint)
        return True
    if "held" in entry and state.discard and not state.outputs_of.get(nid):
        # Discard mode: the earlier attempt held the result on this
        # device under this exact fingerprint. The spool still has it;
        # nothing to write again.
        state.results[nid] = entry["held"]
        state.node_hashes[nid] = resume_mod.content_hash(state.results[nid])
        state.held[nid] = (block, params)
        return True
    return False
