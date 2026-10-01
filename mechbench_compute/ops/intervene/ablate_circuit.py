from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute._mlx import mx
from mechbench_compute.api import (
    DEFAULT_OUTPUT,
    METRICS,
    In,
    Op,
    Output,
    P,
    Resume,
    SpecIntervention,
    collection,
    compile_intervention,
    estimate_bootstrap_ratio,
    find_architecture,
    items_of,
    name_component,
    points,
    read_last_logp,
    read_metric,
    render,
    report_own_top1,
    resolve_target,
    shapes,
)

ABLATIONS = ("zero", "mean")

MEAN_POINTS = ("attn_out", "mlp_out")

OP = Op(
    name="intervene/ablate-circuit",
    needs=frozenset({"model.forward"}),
    resume=Resume("restart"),
    summary="Remove everything outside a circuit, then the circuit itself, and report faithfulness and "
            "completeness.",
    description="""\
The test a circuit owes before anyone believes it. For each record the
metric is read four ways: `m_full`, the model untouched; `m_empty`, every
component of the circuit's universe removed; `m_circuit`, everything in the
universe but the circuit removed, so the circuit runs alone; and
`m_without`, the circuit removed and the rest left alone. Over the records,

- faithfulness = (m̄_circuit − m̄_empty) / (m̄_full − m̄_empty), the share of
  the effect the circuit carries alone, and
- completeness = (m̄_full − m̄_without) / (m̄_full − m̄_empty), the share lost
  when it is removed.

Both are ratios of means, so a record whose full and empty readings nearly
agree cannot make its own ratio arbitrarily large, and neither is clipped:
a circuit whose removal helps the behaviour reads as completeness below 0.
Each has an interval from a bootstrap over records, paired, so a record's
four readings are resampled together.

The full model is read with the same rendering and target as
`intervene/ablate-heads`, so `m_full` is a heads grid's `baseline_logp`
under `logprob`, and a circuit of one head pruned from that grid has
m_full − m_without equal to minus the grid's cell. `m_empty` is shared by
every circuit on the port, which must share one universe. Each removal is
a list of spec items written out component by component — never `except`
— and run as `intervene/apply` runs its spec. That is 2 + 2 × circuits
forward passes per record.

`ablation: "zero"` sets a component's output to zero. `"mean"` replaces it
with its mean over the records given, at that position (every position, for
a component at `all`); it is offered at `attn_out` and `mlp_out` only, since
a per-head mean is not yet possible. A residual location (`resid_*`) is
refused: removing everything else at the residual stream is meaningless, as
each residual already contains the layers before it. Patch it with
`intervene/apply` instead.

The header says whether the records are `held_out`: true when they are not
the records the circuits were found on (their `task.ids`). The `cells`
output holds the four readings per record per circuit.
""",
    inputs=(In("circuit", "intervene/circuit",
               "The circuits to test, from `intervene/prune` or written by hand; they share one universe "
               "and so one `m_empty`.", many=True),
            In("records", "records/record",
               "The prompts the effect is measured on, one per record; a record's prompt is its `user`, "
               "`prompt` or `text` field. A record may carry its own `tracked`."
               " A model reads each record as [its kind](/kinds/records/record/) says: `user` in the chat "
               "template, `prompt` and `text` raw.", many=True),
            In("adapter", "adapter/lora",
               "A LoRA adapter to fuse on top of the model for this node only — "
               "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
               "reference, or a stored adapter. Fuses last, on top of any "
               "adapters the model reference itself carries; `adapter_scale` "
               "scales this one.",
               required=False)),
    output=Output("intervene/faithfulness", collection=True,
                  doc="One item per circuit: `faithfulness` and `completeness` with their intervals, the "
                      "four means, `size` and `n`. The header carries `conditions` (per record `m_full` and "
                      "`m_empty`), `n_off_top1`, `ablation`, `metric`, `universe`, `level`, `resamples`, "
                      "`seed` and `held_out`."),
    outputs={"cells": Output("records/record", collection=True,
                             doc="One item per record per circuit, id `{record}:{circuit}`, with `coords` "
                                 "`{record, circuit}` and the four readings `m_full`, `m_empty`, "
                                 "`m_circuit` and `m_without`.")},
    params=(
        P("ablation", "string",
          "How a component is removed: `zero`, or `mean` over the records given (at `attn_out` and "
          "`mlp_out` only).",
          "zero", choices=ABLATIONS),
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"answer\": \" Paris\"}`; the first is the target, as in "
          "`intervene/ablate-heads`. A record's own `tracked` takes precedence; with none named, the "
          "model's own top-1 prediction is the target.",
          None),
        P("metric", "string",
          "What is read at the decision position: the target's `logprob`, `prob` or `logit`, or "
          "`entropy`, the next-token distribution's entropy in bits.",
          "logprob", choices=METRICS),
        P("level", "float", "The bootstrap interval's level.", 0.95),
        P("resamples", "int", "How many bootstrap resamples, paired over records.", 2000),
    ),
    example={"model": {"$param": "model"}, "ablation": "zero", "metric": "entropy"},
    example_inputs={"circuit": {"$ref": {"bench": "you/lab/circuits"}},
                    "records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    circuits = read_circuits(inputs.get("circuit"))
    records = items_of(inputs.get("records") or [])
    return ablate_circuits(model, circuits, records, params, on_item=ctx.on_item, on_start=ctx.on_start)


def read_circuits(port: Any) -> list[Mapping[str, Any]]:
    if port is None:
        raise ValueError("intervene/ablate-circuit needs circuits on its `circuit` port")
    if isinstance(port, Mapping) and (port.get("kind") == "intervene/circuit" or "components" in port):
        return [port]
    return list(items_of(port))


def read_key(component: Mapping[str, Any]) -> tuple[str, int, int | None, Any]:
    pos = component.get("position", "all")
    head = component.get("head")
    return (str(component.get("point")), int(component["layer"]),
            None if head is None else int(head), "all" if pos in (None, "all") else int(pos))


def list_universe(universe: Mapping[str, Any], n_heads: int) -> list[tuple[str, int, int | None, Any]]:
    positions = universe.get("positions", "all")
    positions = ["all"] if positions in (None, "all") else [int(p) for p in positions]
    heads = int(universe.get("n_heads") or n_heads)
    keys = []
    for point in universe.get("points") or []:
        has_heads = point in points.LAYOUT and points.LAYOUT[point][1] is not None
        for layer in universe.get("layers") or []:
            for head in (range(heads) if has_heads else [None]):
                for pos in positions:
                    keys.append((str(point), int(layer), head, pos))
    return keys


def check_points(model, names: Sequence[str], ablation: str) -> None:
    declared = find_architecture(getattr(model.arch, "model_type", None))
    have = declared.layer_points_of(model.arch) if declared is not None else None
    for point in names:
        if point.startswith("resid_"):
            raise ValueError(
                f"point {point!r}: a residual location is not a component; patch it with intervene/apply")
        if point not in points.LAYOUT or (have is not None and point not in have):
            raise ValueError(
                f"point {point!r} is not one the {getattr(model.arch, 'model_type', '?')!r} forward declares"
                + (f"; it declares {', '.join(have)}" if have is not None else ""))
        if ablation == "mean" and points.LAYOUT[point][1] is not None:
            raise ValueError(
                f"ablation 'mean' at {point!r}: a per-head mean is not available yet (every head would get "
                f"the same mean); use ablation 'zero' there, or mean at {' or '.join(MEAN_POINTS)}")
        if ablation == "mean" and point not in MEAN_POINTS:
            raise ValueError(f"ablation 'mean' is offered at {' and '.join(MEAN_POINTS)}, not {point!r}")


def build_items(keys: Sequence[tuple[str, int, int | None, Any]], ablation: str,
                sources: Mapping[tuple[str, int, Any], Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, Any], list[int]] = {}
    for point, layer, head, pos in keys:
        heads = groups.setdefault((point, layer, pos), [])
        if head is not None:
            heads.append(head)
    items = []
    for (point, layer, pos), heads in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], str(kv[0][2]))):
        item: dict[str, Any] = {"point": point, "layers": [layer],
                                "positions": "all" if pos == "all" else [pos], "op": ablation}
        if heads:
            item["heads"] = sorted(heads)
        if ablation == "mean":
            item["source"] = sources[(point, layer, pos)]
        items.append(item)
    return items


def build_mean_sources(captured: Sequence[Mapping[str, np.ndarray]],
                       groups: Sequence[tuple[str, int, Any]], model_id: str | None) -> dict[tuple[str, int, Any], Any]:
    out = {}
    for point, layer, pos in groups:
        name = f"blocks.{layer}.{point}"
        rows = [acts[name] if pos == "all" else acts[name][[pos]] for acts in captured]
        matrix = np.concatenate(rows, axis=0)
        d = int(matrix.shape[1])
        space = shapes.space(model=model_id, layer=layer, point=point, d=d)
        out[(point, layer, pos)] = collection(
            "activations/vector",
            [{"id": f"mean-{i}", "coords": {}, "space": space, "vector": [float(x) for x in row]}
             for i, row in enumerate(matrix)])
    return out


def ablate_circuits(
    model,
    circuits: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    ablation = str(params.get("ablation") or "zero")
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}: one of {', '.join(ABLATIONS)}")
    metric = str(params.get("metric") or "logprob")
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}: one of {', '.join(METRICS)}")
    level = float(params.get("level", 0.95))
    resamples = int(params.get("resamples", 2000))
    seed = int(params.get("seed", 0))
    if not circuits:
        raise ValueError("intervene/ablate-circuit needs at least one circuit")
    if not records:
        raise ValueError("intervene/ablate-circuit needs at least one record")
    universes = {json.dumps(c.get("universe"), sort_keys=True) for c in circuits}
    if len(universes) > 1:
        raise ValueError(
            f"circuits {[c.get('id') for c in circuits]} come from {len(universes)} universes; they share "
            "m_empty only within one, so test each source's circuits in a node of their own")
    universe = dict(circuits[0].get("universe") or {})
    universe_keys = list_universe(universe, int(model.arch.n_heads))
    if not universe_keys:
        raise ValueError("the circuits' universe names no components: `{points, layers, n_heads?, positions}`")
    check_points(model, sorted({k[0] for k in universe_keys}), ablation)
    inside = set(universe_keys)
    members: list[set] = []
    for c in circuits:
        keys = set()
        for comp in c.get("components") or []:
            key = read_key(comp)
            address = comp.get("address") or name_component(*key)
            if key[0].startswith("resid_"):
                raise ValueError(f"component {address} of {c.get('id')!r}: a residual location is not a "
                                 "component; patch it with intervene/apply")
            if key not in inside:
                raise ValueError(f"component {address} of {c.get('id')!r} is outside the circuit's universe")
            keys.add(key)
        members.append(keys)
    reach = min([k[3] for k in universe_keys if k[3] != "all"], default=0)

    mean_groups = sorted({(p, layer, pos) for p, layer, _h, pos in universe_keys},
                         key=lambda g: (g[1], g[0], str(g[2]))) if ablation == "mean" else []
    captures = sorted({f"blocks.{layer}.{p}" for p, layer, _pos in mean_groups})
    if on_start:
        on_start(len(records) * (2 + 2 * len(circuits)))

    rows: list[dict[str, Any]] = []
    captured: list[dict[str, np.ndarray]] = []
    for record in records:
        r = render(model, record)
        ids = r.array
        n_tokens = int(np.array(ids).shape[-1])
        if -reach > n_tokens:
            raise ValueError(f"record {record.get('id')!r} is {n_tokens} tokens long; the universe reaches "
                             f"position {reach} from the end")
        res = model.run(ids, capture=captures) if captures else model.run(ids)
        lp = read_last_logp(res.logits)
        answer, _ = resolve_target(model, record, params, lp)
        tokens = [model.tokenizer.decode([int(t)]) for t in np.array(ids).reshape(-1)]
        rows.append({"record": record, "ids": ids, "tokens": tokens, "answer": answer, "chat": r.chat,
                     "lp": lp, "m_full": read_metric(answer, metric, res.logits)})
        captured.append({n: np.array(res.cache[n].astype(mx.float32))[0] for n in captures})
        if on_item:
            on_item()

    sources = build_mean_sources(captured, mean_groups, shapes.model_id_of(model)) if mean_groups else {}

    def compile_removal(keys: set | list) -> Any:
        return compile_intervention(model, build_items(sorted(keys, key=str), ablation, sources)) if keys else None

    empty = compile_removal(universe_keys)
    plans = [(compile_removal(inside - keys), compile_removal(keys)) for keys in members]

    def read_removed(row: Mapping[str, Any], compiled: Any) -> float:
        if compiled is None:
            return row["m_full"]
        iv = SpecIntervention(compiled.specs, row["tokens"], row["record"])
        return read_metric(row["answer"], metric, model.run(row["ids"], interventions=[iv]).logits)

    for row in rows:
        row["m_empty"] = read_removed(row, empty)
        if on_item:
            on_item()
        row["circuits"] = []
        for alone, without in plans:
            row["circuits"].append((read_removed(row, alone), read_removed(row, without)))
            if on_item:
                on_item()
                on_item()

    full = np.array([row["m_full"] for row in rows])
    nothing = np.array([row["m_empty"] for row in rows])
    record_ids = [row["record"].get("id") for row in rows]
    items, cells = [], []
    for ci, c in enumerate(circuits):
        kept = np.array([row["circuits"][ci][0] for row in rows])
        without = np.array([row["circuits"][ci][1] for row in rows])
        f_lo, f_hi = estimate_bootstrap_ratio(kept - nothing, full - nothing, level, resamples, seed)
        c_lo, c_hi = estimate_bootstrap_ratio(full - without, full - nothing, level, resamples, seed)
        gap = float(full.mean() - nothing.mean())
        items.append({
            "id": c.get("id"), **({"coords": dict(c["coords"])} if c.get("coords") else {}),
            "circuit": c.get("id"), "size": len(members[ci]), "n": len(rows),
            "faithfulness": round_or_none((kept.mean() - nothing.mean()) / gap if gap else None),
            "completeness": round_or_none((full.mean() - without.mean()) / gap if gap else None),
            "faithfulness_lo": round_or_none(f_lo), "faithfulness_hi": round_or_none(f_hi),
            "completeness_lo": round_or_none(c_lo), "completeness_hi": round_or_none(c_hi),
            "m_full": round(float(full.mean()), 4), "m_circuit": round(float(kept.mean()), 4),
            "m_without": round(float(without.mean()), 4), "m_empty": round(float(nothing.mean()), 4),
        })
        for ri, row in enumerate(rows):
            rid = record_ids[ri]
            cells.append({"id": f"{rid}:{c.get('id')}", "coords": {"record": rid, "circuit": c.get("id")},
                          "m_full": round(float(full[ri]), 6), "m_empty": round(float(nothing[ri]), 6),
                          "m_circuit": round(float(kept[ri]), 6), "m_without": round(float(without[ri]), 6)})
    conditions = []
    for row in rows:
        answer, lp = row["answer"], row["lp"]
        conditions.append({
            "id": row["record"].get("id"),
            "target": shapes.token(model.tokenizer, answer.preferred),
            "variants": answer.variants(model.tokenizer, lp),
            "m_full": round(row["m_full"], 4), "m_empty": round(row["m_empty"], 4),
            "template": "chat" if row["chat"] else "raw",
            **report_own_top1(model, answer, lp),
        })
    held_out = any(set((c.get("task") or {}).get("ids") or []) != set(record_ids) for c in circuits)
    faith = collection(
        "intervene/faithfulness", items,
        conditions=conditions, n_off_top1=sum(1 for c in conditions if "own_top1" in c),
        ablation=ablation, metric=metric, universe=universe, level=level, resamples=resamples, seed=seed,
        held_out=held_out,
        description=(f"Each circuit alone and removed, under {ablation} ablation, read as {metric} at the "
                     f"decision position over {len(rows)} records."))
    return {DEFAULT_OUTPUT: faith,
            "cells": collection("records/record", cells, description=(
                "The four readings per record per circuit: full, empty, the circuit alone, the circuit "
                "removed."))}


def round_or_none(value: Any) -> float | None:
    return None if value is None else round(float(value), 4)
