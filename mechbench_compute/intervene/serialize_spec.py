from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import shapes as S
from mechbench_compute.dictionaries.describe_dictionary import describe_dictionary


def _describe_direction(d: Mapping[str, Any]) -> dict[str, Any]:
    return {"kind": "direction/vector", "space": S.space_of(d), "derivation": d.get("derivation")}


def serialize_spec(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from mechbench_compute.intervene.compile_operator import compile_operator
    from mechbench_compute.lexicon import kinds as K

    out = []
    for it in items:
        w = dict(it)
        for k in ("direction", "direction2"):
            if isinstance(w.get(k), Mapping):
                w[k] = _describe_direction(w[k])
        if w.get("f") is not None:
            w["f"] = compile_operator(str(w["f"])).canonical
        mask = w.get("mask")
        if isinstance(mask, Mapping) and K.item_kind_of(mask) is not None:
            w["mask"] = {"kind": mask.get("kind"), "item_kind": K.item_kind_of(mask),
                         "items": [_describe_direction(d) for d in K.items_of(mask)]}
        elif isinstance(mask, Mapping):
            w["mask"] = _describe_direction(mask)
        elif isinstance(mask, list) and any(isinstance(d, Mapping) for d in mask):
            w["mask"] = [_describe_direction(d) for d in mask]
        if isinstance(w.get("source"), Mapping):
            src = w["source"]
            w["source"] = {"kind": src.get("kind"), "item_kind": K.item_kind_of(src),
                           "n_items": len(K.items_of(src)) if K.item_kind_of(src) else 0}
        if isinstance(w.get("feature"), Mapping) and isinstance(w["feature"].get("dictionary"), Mapping):
            w["feature"] = {**w["feature"], "dictionary": describe_dictionary(w["feature"]["dictionary"])}
        if isinstance(w.get("condition"), Mapping) and isinstance(w["condition"].get("direction"), Mapping):
            w["condition"] = {**w["condition"], "direction": {"derivation": w["condition"]["direction"].get("derivation")}}
        out.append(w)
    return out
