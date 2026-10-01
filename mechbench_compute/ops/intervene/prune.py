from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.api import In, Op, Output, P, Resume, collection, item_kind_of, items_of, name_component

SIGNS = ("negative", "positive", "magnitude")

AGGREGATES = ("mean",)

HEAD_POINT = "attn.per_head_out"

OP = Op(
    name="intervene/prune",
    resume=Resume("restart"),
    summary="Threshold a heads grid or a trace into circuits of the components that carry the effect.",
    description="""\
A heads grid from `intervene/ablate-heads` or a trace from `intervene/patch`
measures an effect at every component of a universe: every head of the
layers it swept, or every (layer, position) at the point it patched. This
op keeps the components that carry most of it and writes them out as an
`intervene/circuit`. It runs no forward pass.

Each component's effect is read in the sense `sign` names: `negative` keeps
components whose removal lowered the metric (the default for a heads grid,
where zeroing a head the answer runs through lowers its log-probability),
`positive` keeps those that raised it (the default for a trace, where
patching the clean activation in recovers the answer), and `magnitude`
keeps either. A component is kept when its effect in that sense is at least
`threshold` times the summed effect, in the same sense, across the
universe; components pointing the other way count as zero in the sum. A
list of thresholds gives one circuit per value. The node's `name` is the
circuit's id, and is required here; with a list of thresholds each circuit
is `{name}@{t}`.
`top` keeps the n largest instead. `per_layer` then keeps at most that
many per layer, the largest first.

Each circuit records `derivation.kept` (how many components), `total` (how
many the universe has) and `kept_share` (the kept components' summed effect
over the universe's). A threshold that keeps nothing gives a circuit with
no components, not no circuit.

A trace is a collection of pairs. Their grids are combined, by `aggregate`,
at the same (layer, position counted from the end); pairs whose prompts
have different lengths do not line up that way and are refused by id, as
is a pair that carries an error instead of a grid. The measure is `share`,
the recovery as a fraction of the pair's clean-minus-corrupt gap, unless
some pair has no gap, in which case every pair is read by `recovery`.

A component's position is counted from the end of the prompt: `-1` is the
last token. A heads grid's components act at every position (`all`). A
trace at a residual point gives residual locations, which say where the
information is carried; `intervene/ablate-circuit` does not remove them.
The circuit's `source.path` is the address of the object on the `grid`
port, when that object was stored.
""",
    inputs=(In("grid", "intervene/heads | intervene/trace",
               "The head-ablation grid or the patching trace to cut.", many=True),),
    output=Output("intervene/circuit", collection=True,
                  doc="One circuit per threshold (or one for `top`), each with `components` (no `links`), "
                      "`metric`, `measure`, `ablation` (`zero` from a heads grid, `patch` from a trace), "
                      "`universe`, `source`, `task` and `derivation`. The header carries `thresholds` when "
                      "there are several."),
    params=(
        P("threshold", "float | list[float]",
          "Keep a component whose effect, in the sign's sense, is at least this fraction of the summed "
          "effect across the universe. A list gives one circuit per value.",
          0.05),
        P("top", "int | null", "Keep the n components with the largest effect instead; not with `threshold`.",
          None),
        P("per_layer", "int | null", "At most this many components per layer, the largest kept, after the "
          "threshold or `top`.", None),
        P("sign", "string | null",
          "Which effects count: `negative`, `positive` or `magnitude`. By default `negative` for a heads "
          "grid and `positive` for a trace.",
          None, choices=SIGNS),
        P("measure", "string | null",
          "Which of the source's measures to threshold. By default `mean_delta` for a heads grid and "
          "`share` for a trace (`recovery` when a pair has no gap).",
          None),
        P("aggregate", "string", "How the pairs of a trace combine at each (layer, position from the end): "
          "`mean`, the only way so far.", "mean"),
    ),
    example={"name": "factual", "threshold": 0.05},
    example_inputs={"grid": {"$ref": {"bench": "you/lab/head-ablation-factual"}}},
)


def run(ctx, inputs, params):
    paths = getattr(ctx, "input_paths", None) or {}
    path = paths.get("grid")
    return prune_circuits(inputs.get("grid"), params,
                          path=path if isinstance(path, str) and path else None)


def prune_circuits(grid: Any, params: Mapping[str, Any], *, path: str | None = None) -> dict[str, Any]:
    if not isinstance(grid, Mapping):
        raise ValueError("intervene/prune needs a heads grid or a trace on its `grid` port")
    name = params.get("name")
    if not name:
        raise ValueError("intervene/prune needs a `name`: the circuit's id")
    if grid.get("kind") == "intervene/heads":
        found = read_heads(grid, params)
    elif item_kind_of(grid) == "intervene/trace":
        found = read_trace(grid, params)
    else:
        raise ValueError(
            f"intervene/prune reads an intervene/heads grid or an intervene/trace collection, not "
            f"{grid.get('item_kind') or grid.get('kind')!r}")
    sign = str(params.get("sign") or found["sign"])
    if sign not in SIGNS:
        raise ValueError(f"unknown sign {sign!r}: one of {', '.join(SIGNS)}")
    top = params.get("top")
    per_layer = params.get("per_layer")
    threshold = params.get("threshold", 0.05)
    if top is not None and "threshold" in params and params["threshold"] is not None:
        raise ValueError("intervene/prune takes `threshold` or `top`, not both")
    cuts: list[tuple[str, dict[str, Any]]]
    if top is not None:
        if int(top) < 0:
            raise ValueError(f"`top` is a count, not {top!r}")
        cuts = [(str(name), {"top": int(top)})]
        thresholds = None
    else:
        thresholds = [float(t) for t in (threshold if isinstance(threshold, (list, tuple)) else [threshold])]
        if not thresholds:
            raise ValueError("`threshold` is a number or a non-empty list of numbers")
        several = isinstance(threshold, (list, tuple))
        cuts = [(f"{name}@{format(t, 'g')}" if several else str(name), {"threshold": t}) for t in thresholds]
        if not several:
            thresholds = None
    cells = found["cells"]
    scores = [score_effect(c["effect"], sign) for c in cells]
    total = sum(s for s in scores if s > 0)
    source = {"path": path, "kind": found["kind"], **found["source"]}
    circuits = []
    for cid, cut in cuts:
        kept = pick_components(cells, scores, total, cut, per_layer)
        share = sum(scores[i] for i in kept) / total if total > 0 else None
        derivation = {"method": "prune", **cut,
                      **({} if per_layer is None else {"per_layer": int(per_layer)}),
                      "sign": sign, "kept": len(kept), "total": len(cells),
                      "kept_share": None if share is None else round(share, 4)}
        circuits.append({
            "id": cid,
            "components": [dict(cells[i]) for i in sorted(kept, key=lambda i: rank_cell(cells[i]))],
            "metric": found["metric"], "measure": found["measure"], "ablation": found["ablation"],
            "universe": found["universe"], "source": source, "task": found["task"],
            "derivation": derivation,
        })
    return collection(
        "intervene/circuit", circuits, model=found.get("model"), thresholds=thresholds,
        description=f"Circuits cut from {found['kind']} by {found['measure']}, {sign} effects.")


def score_effect(effect: float, sign: str) -> float:
    if sign == "negative":
        return -float(effect)
    if sign == "positive":
        return float(effect)
    return abs(float(effect))


def rank_cell(cell: Mapping[str, Any]) -> tuple:
    pos = cell.get("position")
    return (cell["layer"], cell["point"], cell.get("head", -1), -1e9 if pos == "all" else pos)


def pick_components(cells: Sequence[Mapping[str, Any]], scores: Sequence[float], total: float,
                    cut: Mapping[str, Any], per_layer: Any) -> list[int]:
    ranked = sorted((i for i, s in enumerate(scores) if s > 0),
                    key=lambda i: (-scores[i], rank_cell(cells[i])))
    if "top" in cut:
        kept = ranked[:int(cut["top"])]
    else:
        floor = float(cut["threshold"]) * total
        kept = [i for i in ranked if scores[i] >= floor] if total > 0 else []
    if per_layer is not None:
        if int(per_layer) < 0:
            raise ValueError(f"`per_layer` is a count, not {per_layer!r}")
        counts: dict[int, int] = {}
        limited = []
        for i in kept:
            layer = cells[i]["layer"]
            if counts.get(layer, 0) < int(per_layer):
                counts[layer] = counts.get(layer, 0) + 1
                limited.append(i)
        kept = limited
    return kept


def read_heads(grid: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    measure = str(params.get("measure") or "mean_delta")
    measures = grid.get("measures") or {}
    if measure not in measures:
        raise ValueError(f"the heads grid has no measure {measure!r}; it has {sorted(measures)}")
    rows = measures[measure]
    layers = [int(x) for x in grid.get("layers") or []]
    n_heads = int(grid.get("n_heads") or 0)
    if len(rows) != len(layers) or any(len(r) != n_heads for r in rows):
        raise ValueError(f"the heads grid's {measure!r} is not {len(layers)} layers × {n_heads} heads")
    cells = [{"address": name_component(HEAD_POINT, layer, head, "all"), "point": HEAD_POINT,
              "layer": layer, "head": head, "position": "all", "effect": float(rows[li][head])}
             for li, layer in enumerate(layers) for head in range(n_heads)]
    conditions = list(grid.get("conditions") or [])
    task: dict[str, Any] = {"ids": [c.get("id") for c in conditions],
                            "targets": [{"id": c.get("id"), "target": c.get("target")} for c in conditions]}
    if grid.get("n_off_top1") is not None:
        task["n_off_top1"] = int(grid["n_off_top1"])
    return {"kind": "intervene/heads", "cells": cells, "sign": "negative", "measure": measure,
            "metric": str(grid.get("metric") or "logprob"), "ablation": "zero",
            "universe": {"points": [HEAD_POINT], "layers": layers, "n_heads": n_heads, "positions": "all"},
            "source": {}, "task": task, "model": grid.get("model")}


def read_trace(trace: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = str(params.get("aggregate") or "mean")
    if aggregate not in AGGREGATES:
        raise ValueError(f"unknown aggregate {aggregate!r}: one of {', '.join(AGGREGATES)}")
    pairs = items_of(trace)
    if not pairs:
        raise ValueError("the trace has no pairs to prune")
    broken = [p.get("id") for p in pairs if p.get("error") or not p.get("measures")]
    if broken:
        raise ValueError(f"pairs {broken} carry no grid (their errors say why); filter them out first")
    lengths = {p.get("id"): len(p["measures"]["recovery"][0]) for p in pairs}
    first = next(iter(lengths.values()))
    off = [pid for pid, n in lengths.items() if n != first]
    if off:
        raise ValueError(
            f"pairs {off} do not align from the end with {pairs[0].get('id')!r}: their prompts are "
            f"{sorted({lengths[p] for p in off})} tokens long, not {first}; prune them separately")
    named = params.get("measure")
    if named is None:
        measure = "share" if all("share" in p["measures"] for p in pairs) else "recovery"
    else:
        measure = str(named)
        missing = [p.get("id") for p in pairs if measure not in p["measures"]]
        if missing:
            raise ValueError(f"pairs {missing} have no measure {measure!r}")
    layers = [int(x) for x in trace.get("layers") or range(len(pairs[0]["measures"][measure]))]
    point = str(trace.get("point") or "resid_post")
    cells = []
    for li, layer in enumerate(layers):
        for pos in range(first):
            values = [float(p["measures"][measure][li][pos]) for p in pairs]
            cell: dict[str, Any] = {"address": name_component(point, layer, None, pos - first), "point": point,
                                    "layer": layer, "position": pos - first,
                                    "effect": round(sum(values) / len(values), 5)}
            tokens = {str((p.get("tokens") or [None] * first)[pos]) for p in pairs}
            if len(tokens) == 1 and pairs[0].get("tokens"):
                cell["token"] = tokens.pop()
            cells.append(cell)
    task = {"ids": [p.get("id") for p in pairs],
            "targets": [{"id": p.get("id"), "target": p.get("target")} for p in pairs]}
    source = {k: trace[k] for k in ("method", "point") if trace.get(k) is not None}
    return {"kind": "intervene/trace", "cells": cells, "sign": "positive", "measure": measure,
            "metric": str(trace.get("metric") or "logprob"), "ablation": "patch",
            "universe": {"points": [point], "layers": layers, "positions": list(range(-first, 0))},
            "source": source, "task": task, "model": trace.get("model")}
