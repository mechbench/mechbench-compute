from __future__ import annotations


def read_resume_entry(state, nid: str, fingerprint: str):
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
