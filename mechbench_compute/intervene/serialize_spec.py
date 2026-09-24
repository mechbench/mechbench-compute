from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import shapes as S


def serialize_spec(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        w = dict(it)
        for k in ("direction", "direction2"):
            if isinstance(w.get(k), Mapping):
                w[k] = {"kind": "direction/vector", "space": S.space_of(w[k]),
                        "derivation": w[k].get("derivation")}
        if isinstance(w.get("source"), Mapping):
            from mechbench_compute.lexicon import kinds as K

            src = w["source"]
            w["source"] = {"kind": src.get("kind"), "item_kind": K.item_kind_of(src),
                           "n_items": len(K.items_of(src)) if K.item_kind_of(src) else 0}
        if isinstance(w.get("condition"), Mapping) and isinstance(w["condition"].get("direction"), Mapping):
            w["condition"] = {**w["condition"], "direction": {"derivation": w["condition"]["direction"].get("derivation")}}
        out.append(w)
    return out
