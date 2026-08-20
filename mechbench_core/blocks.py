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


def _factor_levels(factor: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Materialize a factor to its [{key, value, coords}] levels. A
    factor may carry enumerated `levels`, `sampled` generators (one or
    a list — the Marcus seed factor uses five), or both (enumerated
    first). Levels and generators may attach extra `coords` merged
    into each record (generators stamp a `<name>_kind` coordinate by
    default, so analysis groups by generator type, never by parsing
    keys). (`values` accepted as a legacy spelling of `levels`.)"""
    name = factor.get("name", "")
    out: list[dict[str, Any]] = []
    enumerated = factor.get("levels", factor.get("values"))
    if enumerated:
        out += [{"key": v["key"], "value": v.get("value", v["key"]),
                 "coords": dict(v.get("coords", {}))}
                for v in enumerated]
    sampled = factor.get("sampled")
    if sampled:
        gens = sampled if isinstance(sampled, list) else [sampled]
        for gen in gens:
            start = int(gen.get("start", 0))
            count = int(gen["count"])
            prefix = gen.get("key_prefix") or f"{gen['kind']}-{gen['size']}"
            kind_coord = gen.get("kind_coord", f"{name}_kind")
            extra = {kind_coord: prefix, **dict(gen.get("coords", {}))}
            out += [{"key": f"{prefix}-i{i}",
                     "value": _sample_value(gen, i),
                     "coords": dict(extra)}
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
                    "coords": {**rec["coords"], name: v["key"],
                               **v.get("coords", {})},
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
            # Fixpoint substitution (bounded): a level's text may itself
            # contain placeholders (the Marcus elaborate opening embeds
            # {gender}) — passes repeat while substitutions still fire.
            for _ in range(4):
                before = s
                for axis, value in rec.get("values", {}).items():
                    s = s.replace("{" + axis + "}", str(value))
                if s == before:
                    break
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

def suite_metric_records(results: Mapping[str, Any],
                         n_samples: Mapping[str, Any] | None = None,
                         variant: str = "base") -> list[dict[str, Any]]:
    """Shape lm-eval-harness `results` (task -> {"acc,none": v,
    "acc_stderr,none": s, ...}) into coord-carrying records:
    one record per (task, metric) with value/stderr/n and coords
    {task, metric, variant} — composable straight into union /
    paired_delta for base-vs-adapter deltas."""
    out: list[dict[str, Any]] = []
    for task in sorted(results):
        metrics = results[task]
        ns = (n_samples or {}).get(task) or {}
        n = ns.get("effective", ns.get("original"))
        stderrs = {}
        values = {}
        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            name = key.split(",", 1)[0]
            if name in ("sample_len",):  # harness bookkeeping, not a metric
                continue
            if name.endswith("_stderr"):
                stderrs[name[: -len("_stderr")]] = float(val)
            else:
                values[name] = float(val)
        for name in sorted(values):
            rec: dict[str, Any] = {
                "id": f"{task}:{name}:{variant}",
                "coords": {"task": task, "metric": name,
                           "variant": variant},
                "value": values[name],
            }
            if name in stderrs:
                rec["stderr"] = stderrs[name]
            if n is not None:
                rec["n"] = int(n)
            out.append(rec)
    return out


def table_from_records(records: Any,
                       params: Mapping[str, Any]) -> dict[str, Any]:
    """Present a record stream as a metric table: coords flatten into
    leading columns, remaining scalar fields follow. The generic
    records -> table presenter (delta tables, group stats, ...)."""
    recs = _records(records)
    coord_keys: list[str] = []
    value_keys: list[str] = []
    for r in recs:
        for k in r.get("coords", {}):
            if k not in coord_keys:
                coord_keys.append(k)
        for k, v in r.items():
            if k in ("id", "coords") or not isinstance(v, (int, float, str)):
                continue
            if k not in value_keys:
                value_keys.append(k)
    rows = []
    for r in recs:
        row: dict[str, Any] = {"id": r.get("id")}
        row.update({k: r.get("coords", {}).get(k) for k in coord_keys})
        row.update({k: r.get(k) for k in value_keys if k in r})
        rows.append(row)
    dtypes = {}
    for k in ["id", *coord_keys, *value_keys]:
        vals = [row.get(k) for row in rows if row.get(k) is not None]
        dtypes[k] = ("number" if vals and all(
            isinstance(v, (int, float)) for v in vals) else "string")
    return {"kind": "metric_table",
            "name": params.get("name", "records"),
            "description": params.get("description", ""),
            "row_axis": params.get("row_axis", "record"),
            "columns": [{"name": k, "dtype": d} for k, d in dtypes.items()],
            "rows": rows}


def chart_spec(records: Any, params: Mapping[str, Any],
               source_label: str | None = None) -> dict[str, Any]:
    """~canonical/ops/chart/spec/1 — a chart is a bench object (task
    000277): a presentation spec over the upstream table/records. When
    the executor knows the input's label the spec REFERENCES it
    (`source`, lineage-true, renders live); otherwise the rows ride
    inline (`data.rows`) so the chart stays self-contained."""
    enc = params.get("encoding") or {}
    x = enc.get("x") or params.get("x")
    y = enc.get("y") or params.get("y")
    if not x or not y:
        raise ValueError("chart/spec needs encoding.x and encoding.y")
    spec: dict[str, Any] = {
        "kind": "chart_spec",
        "title": params.get("title", ""),
        "mark": params.get("mark", "bar"),
        "encoding": {"x": x, "y": y,
                     **({"series": enc["series"]} if enc.get("series") else {})},
    }
    if source_label:
        spec["source"] = source_label
    else:
        recs = _records(records) if not (isinstance(records, Mapping)
                                         and isinstance(records.get("rows"), list)) \
            else records["rows"]
        rows = []
        for r in recs:
            row = {k: v for k, v in r.items() if k != "coords"}
            row.update(r.get("coords", {}) if isinstance(r, Mapping) else {})
            rows.append(row)
        spec["data"] = {"rows": rows}
    return spec

PURE_BLOCKS: dict[str, Callable[..., Any]] = {
    "~canonical/ops/factor-cross/1":
        lambda inputs, params: factor_cross(params),
    # Alias: protocol versions pinned before the rename still execute.
    "~canonical/ops/grid/1":
        lambda inputs, params: factor_cross(params),
    "~canonical/ops/template/1":
        lambda inputs, params: template(
            _records(inputs.get("records") or params.get("records")),
            params),
    "~canonical/ops/select/1":
        lambda inputs, params: select(inputs["records"], params),
    "~canonical/ops/paired-delta/1":
        lambda inputs, params: paired_delta(inputs["records"], params),
    "~canonical/ops/group-stats/1":
        lambda inputs, params: group_stats(inputs["records"], params),
    "~canonical/ops/table/from-records/1":
        lambda inputs, params: table_from_records(
            inputs.get("records") or params.get("records"), params),
    "~canonical/ops/union/1":
        lambda inputs, params: union(inputs, params),
    "~canonical/ops/eval/expectation/1":
        lambda inputs, params: eval_expectation(inputs, params),
}


def eval_expectation(inputs: Mapping[str, Any],
                     params: Mapping[str, Any]) -> dict[str, Any]:
    """The first member of the eval block family (~canonical/ops/eval/):
    judge decision-read results against per-condition EXPECTATIONS
    carried as data, publishing a metric table with verdicts.

    Expectation kinds (per record, joined on id):
      {"kind": "uniform", "over": [outcomes], "max_kl_bits": t}
          -> kl_bits from uniform over the outcome masses; pass iff
             kl_bits <= t and the outcomes carry real mass.
      {"kind": "answer", "value": tok, "min_p": t}
          -> p_expected from the read's top tokens; pass iff >= t.
      {"kind": "min_entropy", "bits": t}
          -> pass iff the decision entropy >= t (diversity floor).

    The aggregate row (id "ALL") carries the pass rate — the number a
    publication cites.
    """
    import math

    results = _records(inputs.get("results") or params.get("results"))
    expectations = {r["id"]: r["expect"]
                    for r in _records(inputs.get("expectations")
                                       or params.get("expectations"))}
    rows = []
    n_pass = 0
    n_judged = 0
    for c in results:
        exp = expectations.get(c["id"])
        if not exp:
            continue
        row: dict[str, Any] = {"id": c["id"], "expect": exp["kind"],
                                "entropy_bits": c.get("entropy_bits")}
        ok = False
        if exp["kind"] == "uniform":
            masses = c.get("outcome_mass") or {}
            over = exp["over"]
            ps = [float(masses.get(str(o), 0.0)) for o in over]
            tot = sum(ps)
            if tot > 0:
                kl = sum(q / tot * math.log2((q / tot) / (1.0 / len(over)))
                         for q in ps if q > 0)
                row["kl_bits"] = round(kl, 4)
                row["outcome_mass"] = round(tot, 4)
                ok = kl <= float(exp.get("max_kl_bits", 0.1))
        elif exp["kind"] == "answer":
            want = str(exp["value"])
            p = None
            for t in c.get("top_tokens") or []:
                if t["token"] == want:
                    p = float(t["p"])
                    break
            row["p_expected"] = round(p, 4) if p is not None else None
            ok = p is not None and p >= float(exp.get("min_p", 0.99))
        elif exp["kind"] == "min_entropy":
            ok = float(c.get("entropy_bits") or 0.0) >= float(exp["bits"])
        else:
            raise ValueError(f"unknown expectation kind: {exp['kind']!r}")
        row["pass"] = ok
        n_judged += 1
        n_pass += int(ok)
        rows.append(row)
    rows.append({"id": "ALL", "expect": "aggregate",
                 "pass_rate": round(n_pass / n_judged, 4) if n_judged else None,
                 "n_pass": n_pass, "n_judged": n_judged})
    cols = {"id": "string", "expect": "string", "entropy_bits": "number",
            "kl_bits": "number", "outcome_mass": "number",
            "p_expected": "number", "pass": "string",
            "pass_rate": "number", "n_pass": "number", "n_judged": "number"}
    return {"kind": "metric_table",
            "name": params.get("name", "expectation-eval"),
            "description": params.get("description", ""),
            "row_axis": "condition",
            "columns": [{"name": k, "dtype": d} for k, d in cols.items()],
            "rows": [{k: (str(v) if k == "pass" else v)
                       for k, v in r.items()} for r in rows]}
