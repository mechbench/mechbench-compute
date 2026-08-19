"""Stdlib building blocks (epic 000258, arc B): the pure blocks —
Grid and Template — plus the block registry the pipeline executor
resolves op refs against.

Design rules these implement:

- **FactorCross** (nee Grid): factors -> records with coordinates —
  the fully crossed design of experimental methodology. A factor
  enumerates its levels or samples them from a seeded generator.
  Records carry {id, coords, values}: `coords` are the level KEYS
  (structured, for GroupBy/PairedDelta — never parsed from the id),
  `values` the substitution payloads (which carry their own
  whitespace; there is no hidden joining logic). "Axis" is reserved
  for the charting surface, deliberately.
- **Range-splitting invariance**: sampled axes derive one rng per
  instance from (seed, index), so generate(seed, 0, 1000) equals
  generate(seed, 0, 100) + generate(seed, 100, 900). Incremental
  dataset growth is the same node run over a later range.
- **Template**: records x named string templates -> the same records
  with instantiated string fields. Source-agnostic: records may come
  from a Grid or from any record stream (dataset rows, later).
  No block in this module knows what a "prompt" is.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping

# The original ai-randomness noise charset, reproduced exactly.
SEED_CHARS = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "1234567890!@#$%^&*-_=+`~[]{}\\|;'\"/?.>,<")


def _sample_value(gen: Mapping[str, Any], index: int) -> str:
    """One sampled axis value, deterministic in (seed, index) alone —
    the range-splitting guarantee."""
    kind = gen["kind"]
    size = int(gen["size"])
    rng = random.Random(f"{gen.get('seed', 0)}:{index}")
    if kind == "noise":
        body = "".join(rng.choice(SEED_CHARS) for _ in range(size))
    elif kind == "words":
        # word_list: an inline list, or a fetched word_list object
        # payload ({kind: "word_list", words: [...]}) — the executor's
        # {"$fetch": ref} resolution hands the payload through whole.
        raw = gen.get("word_list") or []
        words = raw.get("words") if isinstance(raw, Mapping) else raw
        if not words:
            raise ValueError(
                "words generator needs word_list (inline list or a "
                "fetched word_list object)")
        body = " ".join(rng.choice(words) for _ in range(size))
    else:
        raise ValueError(f"unknown generator kind: {kind!r}")
    wrap = gen.get("wrap", "{x}")
    return wrap.replace("{x}", body)


def _factor_levels(factor: Mapping[str, Any]) -> list[dict[str, str]]:
    """Materialize a factor to its [{key, value}] levels. A factor may
    carry enumerated `levels`, a `sampled` generator, or both
    (enumerated first) — the Marcus seed factor is `none` plus sampled
    instances. (`values` accepted as a legacy spelling of `levels`.)"""
    out: list[dict[str, str]] = []
    enumerated = factor.get("levels", factor.get("values"))
    if enumerated:
        out += [{"key": v["key"], "value": v.get("value", v["key"])}
                for v in enumerated]
    if "sampled" in factor:
        gen = factor["sampled"]
        start = int(gen.get("start", 0))
        count = int(gen["count"])
        prefix = gen.get("key_prefix") or f"{gen['kind']}-{gen['size']}"
        out += [{"key": f"{prefix}-i{i}", "value": _sample_value(gen, i)}
                for i in range(start, start + count)]
    if not out:
        raise ValueError(
            f"factor {factor.get('name')!r} has neither levels nor sampled")
    return out


def factor_cross(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fully crossed factors: the Cartesian product of every factor's
    levels, as coordinate-carrying records. (`axes` accepted as a
    legacy spelling of `factors` for pre-rename protocol versions.)"""
    factors = params.get("factors", params.get("axes")) or []
    records: list[dict[str, Any]] = [
        {"id": "", "coords": {}, "values": {}}
    ]
    for axis in factors:
        name = axis["name"]
        vals = _factor_levels(axis)
        nxt = []
        for rec in records:
            for v in vals:
                nxt.append({
                    "id": "-".join(
                        [p for p in [rec["id"], v["key"]] if p]),
                    "coords": {**rec["coords"], name: v["key"]},
                    "values": {**rec["values"], name: v["value"]},
                })
        records = nxt
    for i, rec in enumerate(records):
        if not rec["id"]:
            rec["id"] = f"record-{i}"
    return records


def template(records: list[dict[str, Any]],
             params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Instantiate named string templates against each record's values.
    `{axis-name}` placeholders substitute; everything else is verbatim."""
    templates: Mapping[str, str] = params.get("templates") or {}
    out = []
    for rec in records:
        fields = {}
        for fname, tmpl in templates.items():
            s = str(tmpl)
            for axis, value in rec.get("values", {}).items():
                s = s.replace("{" + axis + "}", str(value))
            fields[fname] = s
        out.append({"id": rec["id"], "coords": dict(rec.get("coords", {})),
                    **fields})
    return out


def _records(x: Any) -> list[dict[str, Any]]:
    """Coerce a node output to its record list: blocks pass bare lists
    or dicts wrapping them under a conventional key."""
    if isinstance(x, list):
        return x
    if isinstance(x, Mapping):
        for k in ("records", "conditions", "rows"):
            if isinstance(x.get(k), list):
                return x[k]
    raise ValueError("input is not a record stream")


def select(records: Any, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Filter records by coords equality; optionally project fields.
    where: {coord: value | [values]}; fields: [names] keeps id+coords
    plus the named fields."""
    recs = _records(records)
    where: Mapping[str, Any] = params.get("where") or {}
    out = []
    for r in recs:
        coords = r.get("coords", {})
        ok = all(
            coords.get(k) in (v if isinstance(v, list) else [v])
            for k, v in where.items()
        )
        if not ok:
            continue
        fields = params.get("fields")
        if fields:
            out.append({"id": r.get("id"), "coords": dict(coords),
                        **{f: r.get(f) for f in fields}})
        else:
            out.append(r)
    return out


def paired_delta(records: Any, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """For each non-baseline record, subtract its matched baseline's
    value. match_on: coords that must agree; baseline_where: coords
    identifying the baseline records; value: the numeric field.
    Output records keep coords (minus nothing) plus value/baseline/
    delta fields — composable straight into group_stats."""
    recs = _records(records)
    match_on = params.get("match_on") or []
    baseline_where: Mapping[str, Any] = params["baseline_where"]
    value_field = params["value"]

    def is_baseline(r):
        return all(r.get("coords", {}).get(k) == v
                   for k, v in baseline_where.items())

    baselines = {}
    for r in recs:
        if is_baseline(r):
            key = tuple(r.get("coords", {}).get(k) for k in match_on)
            baselines[key] = r
    out = []
    for r in recs:
        if is_baseline(r):
            continue
        key = tuple(r.get("coords", {}).get(k) for k in match_on)
        base = baselines.get(key)
        if base is None:
            raise ValueError(f"no baseline for record {r.get('id')!r}")
        v, b = float(r[value_field]), float(base[value_field])
        out.append({"id": r.get("id"), "coords": dict(r.get("coords", {})),
                    "value": v, "baseline": b,
                    "delta": round(v - b, 6)})
    return out


def group_stats(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """Group records by coords and summarize a numeric field into
    MetricTable-shaped rows. by: [coord names] ([] = one overall
    group); value: field name; stats fixed: n/median/mean/min/max +
    share_negative (useful for deltas)."""
    from statistics import mean, median
    recs = _records(records)
    by = params.get("by") or []
    value_field = params["value"]
    groups: dict[tuple, list[float]] = {}
    for r in recs:
        key = tuple(r.get("coords", {}).get(k) for k in by)
        groups.setdefault(key, []).append(float(r[value_field]))
    rows = []
    for key, vals in groups.items():
        row = {k: key[i] for i, k in enumerate(by)}
        row.update({
            "n": len(vals),
            "median": round(median(vals), 4),
            "mean": round(mean(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "share_negative": round(sum(v < 0 for v in vals) / len(vals), 3),
        })
        rows.append(row)
    columns = [{"name": k, "dtype": "string"} for k in by] + [
        {"name": n, "dtype": "number"}
        for n in ("n", "median", "mean", "min", "max", "share_negative")
    ]
    return {"kind": "metric_table",
            "name": params.get("name", f"{value_field}-stats"),
            "description": params.get("description", ""),
            "row_axis": "condition", "columns": columns, "rows": rows}


def union(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Concatenate record streams, structurally recording source
    segments; each record gains a batch coordinate named after its
    input port. Collections never mutate — growth is union."""
    batch_axis = params.get("batch_axis", "batch")
    segments = []
    records = []
    for port in sorted(inputs.keys()):
        recs = _records(inputs[port])
        segments.append({"source": port, "count": len(recs)})
        for r in recs:
            records.append({**r, "coords": {**r.get("coords", {}),
                                            batch_axis: port}})
    return {"kind": "record_set", "segments": segments, "records": records}


# --- registry ---------------------------------------------------------------

# Op ref -> callable. Pure blocks take (inputs, params); model blocks
# are registered by the executor host (the runner), which owns model
# lifecycle. Descriptor objects at ~canonical/ops/... arrive with the
# 000248 registry arc; until then this in-code table is the resolver.
PURE_BLOCKS: dict[str, Callable[..., Any]] = {
    "~canonical/ops/factor-cross/1":
        lambda inputs, params: factor_cross(params),
    # Alias: protocol versions pinned before the rename still execute.
    "~canonical/ops/grid/1":
        lambda inputs, params: factor_cross(params),
    "~canonical/ops/template/1":
        lambda inputs, params: template(_records(inputs["records"]), params),
    "~canonical/ops/select/1":
        lambda inputs, params: select(inputs["records"], params),
    "~canonical/ops/paired-delta/1":
        lambda inputs, params: paired_delta(inputs["records"], params),
    "~canonical/ops/group-stats/1":
        lambda inputs, params: group_stats(inputs["records"], params),
    "~canonical/ops/union/1":
        lambda inputs, params: union(inputs, params),
}
