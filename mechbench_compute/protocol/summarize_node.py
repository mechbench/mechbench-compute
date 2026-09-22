"""What a node produced, small enough to read beside the node."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import lexicon


def summarize_node(value: Any, spend: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What a node produced, in the terms a reader asks first: which kind,
    and how many. `{kind, collection, items}` for a collection (however it
    is spelled — a retired plural object, a bare list); `{kind,
    collection: false}` for one object, with `rows` when it is a table of
    them; `{}` for a value that carries no kind. `spend_usd` when the node
    called a provider."""
    out: dict[str, Any] = {}
    if isinstance(value, list):
        out = {"kind": lexicon.COLLECTION, "collection": True, "items": len(value)}
    elif isinstance(value, Mapping):
        item_kind = lexicon.item_kind_of(value)
        if item_kind is not None:
            out = {"kind": item_kind, "collection": True, "items": len(lexicon.items_of(value))}
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
