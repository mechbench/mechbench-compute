"""What a previous attempt left for one node, when it may still be
used."""

from __future__ import annotations


def read_resume_entry(state, nid: str, fingerprint: str):
    """The resume map's entry for this node, or `None` when it cannot
    stand: the node's process identity has changed since it was
    written, or a consumer requires more than this block offers, and
    either way the node runs again from the start."""
    entry = state.resume.get(nid) if isinstance(state.resume, dict) else None
    if not entry:
        return None
    if entry.get("fingerprint") != fingerprint:
        print(f"[resume] {nid}: fingerprint changed; restarting")
        return None
    if nid in state.forced_restart:
        print(f"[resume] {nid}: a consumer requires more than "
              f"this block offers; restarting")
        return None
    return entry
