from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import lexicon


def summarize_node(value: Any, spend: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, list):
        out = {"kind": lexicon.COLLECTION, "collection": True, "items": len(value)}
    elif isinstance(value, Mapping):
        item_kind = lexicon.item_kind_of(value)
        if item_kind is not None:
            out = {"kind": item_kind, "collection": True, "items": len(lexicon.items_of(value))}
            if isinstance(value.get("ended"), Mapping):
                out["ended"] = dict(value["ended"])
        elif isinstance(value.get("kind"), str):
            try:
                name, _plural = lexicon.resolve_kind(value["kind"], warn=False)
            except KeyError:
                name = value["kind"]
            out = {"kind": name, "collection": False}
            if isinstance(value.get("rows"), list):
                out["rows"] = len(value["rows"])
    if spend and spend.get("cost_usd") is not None:
        out["spend_usd"] = round(float(spend["cost_usd"]), 8)
    return out
