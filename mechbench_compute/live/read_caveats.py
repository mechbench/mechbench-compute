from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

FEW = 8


def read_caveats(items: Sequence[Mapping[str, Any]], header: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    saturated = sum(1 for it in items if isinstance(it.get("tracked"), Mapping)
                    for t in it["tracked"].values() if isinstance(t, Mapping) and t.get("saturated") is True)
    if saturated:
        out.append({"code": "SATURATED", "count": saturated})
    cut = sum(1 for it in items if is_cut(it))
    if cut:
        out.append({"code": "CUT", "count": cut})
    off = count_off_top1(items, header)
    if off:
        out.append({"code": "OFF_TOP1", "count": off})
    under = sum(1 for it in items if is_under_majority(it))
    if under:
        out.append({"code": "UNDER_MAJORITY", "count": under})
    n, unit = count_items(items)
    if 0 < n < FEW:
        out.append({"code": "FEW_ITEMS", "count": n, "unit": unit, "fewest": FEW})
    return out


def is_cut(item: Mapping[str, Any]) -> bool:
    meta = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    if (meta.get("sampling") or {}).get("ended") == "max_tokens":
        return True
    return any(is_truncated(where.get("truncated")) for where in (item, meta))


def is_truncated(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, int) and value > 0


def count_off_top1(items: Sequence[Mapping[str, Any]], header: Mapping[str, Any]) -> int:
    said = header.get("n_off_top1")
    if isinstance(said, int) and not isinstance(said, bool):
        return said
    own = [it.get("n_off_top1") for it in items]
    if any(isinstance(n, int) and not isinstance(n, bool) for n in own):
        return sum(n for n in own if isinstance(n, int) and not isinstance(n, bool))
    return sum(1 for it in items if it.get("own_top1"))


def is_under_majority(item: Mapping[str, Any]) -> bool:
    over = item.get("over_baseline")
    if isinstance(over, (int, float)) and not isinstance(over, bool):
        return over < 0
    accuracy, majority = item.get("accuracy_test"), item.get("baseline")
    return (isinstance(accuracy, (int, float)) and isinstance(majority, (int, float))
            and not isinstance(accuracy, bool) and accuracy < majority)


def count_items(items: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    sizes = [it.get("n_items") for it in items if isinstance(it.get("n_items"), int)]
    if sizes:
        return min(sizes), "item"
    ids = {it.get("id") for it in items if it.get("id") is not None}
    return (len(ids), "record") if ids else (len(items), "item")
