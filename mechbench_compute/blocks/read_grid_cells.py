from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.intervene.name_component import name_component

HEAD_POINT = "attn.per_head_out"
BLOCK_POINT = "block"
CELL_KINDS = ("intervene/heads", "intervene/trace", "logits/lens", "logits/attribution",
              "activations/divergence")


def read_grid_cells(grid: Mapping[str, Any], kind: str | None,
                    header: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    if kind not in CELL_KINDS:
        return None
    measures = grid.get("measures")
    if grid.get("error") or not isinstance(measures, Mapping) or not measures:
        return []
    names = [str(n) for n in measures]
    if kind == "logits/attribution":
        return _read_components(grid, header, names)
    rows = measures[names[0]]
    layers = [int(x) for x in (grid.get("layers") or header.get("layers") or range(len(rows)))]
    if len(layers) != len(rows):
        raise ValueError(f"grid {grid.get('id')!r} has {len(rows)} rows but names {len(layers)} layers")
    out: list[dict[str, Any]] = []
    if kind == "intervene/heads":
        for li, layer in enumerate(layers):
            for head in range(len(rows[li])):
                cell = {"address": name_component(HEAD_POINT, layer, head, "all"), "point": HEAD_POINT,
                        "layer": layer, "head": head, "position": "all"}
                out.append(_measure(cell, measures, names, (li, head)))
        return out
    point = str(header.get("point") or "resid_post")
    tokens = grid.get("tokens") if isinstance(grid.get("tokens"), list) else None
    for li, layer in enumerate(layers):
        width = len(rows[li])
        for pos in range(width):
            cell = {"address": name_component(point, layer, None, pos - width), "point": point,
                    "layer": layer, "position": pos - width}
            if tokens is not None and len(tokens) == width:
                cell["token"] = tokens[pos]
            out.append(_measure(cell, measures, names, (li, pos)))
    return out


def _read_components(grid: Mapping[str, Any], header: Mapping[str, Any], names: list[str]) -> list[dict[str, Any]]:
    measures = grid["measures"]
    values = measures[names[0]]
    components = [str(c) for c in (header.get("components") or [])]
    if len(components) != len(values):
        raise ValueError(f"attribution {grid.get('id')!r} has {len(values)} values but the header names "
                         f"{len(components)} components")
    layers = [int(c[1:]) for c in components if c != "embed"]
    out = []
    for i, component in enumerate(components):
        if component == "embed":
            layer, point = (layers[0] if layers else 0), "resid_pre"
        else:
            layer, point = int(component[1:]), BLOCK_POINT
        cell = {"address": name_component(point, layer, None, -1), "point": point, "layer": layer,
                "position": -1, "component": component}
        out.append(_measure(cell, measures, names, (i,)))
    return out


def _measure(cell: dict[str, Any], measures: Mapping[str, Any], names: list[str],
             index: tuple[int, ...]) -> dict[str, Any]:
    for name in names:
        v: Any = measures[name]
        for i in index:
            v = v[i] if isinstance(v, list) and i < len(v) else None
        cell[name] = v
    return cell
