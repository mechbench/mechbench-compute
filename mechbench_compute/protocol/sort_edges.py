from __future__ import annotations

from typing import Any


def sort_edges(edges, nid: str) -> list[Any]:
    into = [e for e in edges if (e.get("to") or {}).get("node") == nid]
    return sorted(into, key=lambda e: (str(e["to"].get("port", "")),
                                       int(e.get("index", 0)),
                                       str((e.get("from") or {}).get("node", ""))))
