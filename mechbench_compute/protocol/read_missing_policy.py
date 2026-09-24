from __future__ import annotations


def read_missing_policy(decl, edges) -> str:
    chosen = [e.get("on_missing") for e in edges if e.get("on_missing")]
    if chosen:
        return str(chosen[0])
    return decl.on_missing if decl is not None else "fail"
