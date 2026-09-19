"""Stdlib building blocks (epic 000258, arc B): the pure blocks, plus
the block registry the pipeline executor resolves op refs against.

An operation is named once, in the lexicon, and that name is used
everywhere — here, in a stored graph, on the docs page. This docstring
called these `Grid`, `Template`, `FactorCross` and `PairedDelta` until
2026-09-17, names retired by 000495 and 000510 and carried nowhere
else; teaching a reader a vocabulary the executor would refuse is the
same fault as shipping one.

Design rules these implement:

- **`records/cross`**: factors -> records with coordinates —
  the fully crossed design of experimental methodology. A factor
  enumerates its levels or samples them from a seeded generator.
  Records carry {id, coords, values}: `coords` are the level KEYS
  (structured, for `records/summarize` and `records/subtract` — never
  parsed from the id), `values` the substitution payloads (which carry
  their own whitespace; there is no hidden joining logic). "Axis" is
  reserved for the charting surface, deliberately.
- **Range-splitting invariance**: sampled axes derive one rng per
  instance from (seed, index), so generate(seed, 0, 1000) equals
  generate(seed, 0, 100) + generate(seed, 100, 900). Incremental
  dataset growth is the same node run over a later range.
- **`records/fill`**: records x named string templates -> the same
  records with instantiated string fields. Source-agnostic: records may
  come from `records/cross` or from any record stream (dataset rows,
  later).
  No block in this module knows what a "prompt" is.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# The original ai-randomness noise charset, reproduced exactly.
SEED_CHARS = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "1234567890!@#$%^&*-_=+`~[]{}\\|;'\"/?.>,<")


def _sample_value(gen: Mapping[str, Any], index: int) -> str:
    """One sampled axis value, deterministic in (seed, index) alone —
    the range-splitting guarantee."""
    kind = gen.get("type") or gen["kind"]
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
            prefix = gen.get("key_prefix") or f"{gen.get('type') or gen['kind']}-{gen['size']}"
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


def _items(x: Any) -> list[dict[str, Any]]:
    """The items of an input, however it arrived — the lexicon's one
    reader, for a `collection`, a bare list, or an older plural object."""
    from mechbench_compute.lexicon import kinds as K

    return K.items_of(x)


def _pop_path(rec: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Remove the value at a dotted path, copying each container on the
    way so the input record is never mutated. (found, value)."""
    parts = path.split(".")
    cur = rec
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, Mapping):
            return False, None
        nxt = dict(nxt)
        cur[p] = nxt
        cur = nxt
    if parts[-1] not in cur:
        return False, None
    return True, cur.pop(parts[-1])


def _set_path(rec: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = rec
    for p in parts[:-1]:
        nxt = cur.get(p)
        nxt = dict(nxt) if isinstance(nxt, Mapping) else {}
        cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def rename(records: Any, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """records/rename: move fields on every record, `fields: {old: new}`.
    A name may be a dotted path (`coords.opening`, `metadata.coords`), so
    a value moves into or out of a nested object. A record without the
    old field is left as it is; everything not named is kept."""
    fields = params.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("records/rename needs `fields`: {\"old\": \"new\", …}")
    out = []
    for r in _items(records):
        rec = dict(r)
        for old, new in fields.items():
            found, value = _pop_path(rec, str(old))
            if found:
                _set_path(rec, str(new), value)
        out.append(rec)
    return out


def _coll(items: list[dict[str, Any]], **header: Any) -> dict[str, Any]:
    """A `collection` of `records/record`."""
    from mechbench_compute.lexicon import kinds as K

    return K.collection("records/record", items, **header)


def select(records: Any, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Filter records by equality; optionally project fields.
    where: {key: value | [values]} — a key is read from `coords` when it
    is a coord, and from the record itself otherwise, so a field that
    `text/measure` `annotate` wrote (a pattern hit is a field, not a
    coord) filters too (task 000368). fields: [names] keeps id+coords
    plus the named fields."""
    recs = _items(records)
    where: Mapping[str, Any] = params.get("where") or {}
    out = []
    for r in recs:
        coords = r.get("coords", {})
        ok = all(
            (coords.get(k) if k in coords else r.get(k))
            in (v if isinstance(v, list) else [v])
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
    recs = _items(records)
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


def _group_key(record: Mapping[str, Any], by: Sequence[str]) -> tuple:
    """The grouping key, read from the record's coordinates and then from
    the record itself.

    A coordinate is where a condition belongs, and most ops put it
    there. Some write the varying thing at the top level instead —
    `intervene/ablate-layers` emits `{id, layer, delta_logp}`, the layer
    being exactly the condition — and grouping by `layer` then silently
    produced ONE row keyed `None` instead of forty-two. The VALUE was
    already read from the top level, so the asymmetry was the bug: a
    field is a field wherever the record carries it (task 000590).
    """
    coords = record.get("coords") or {}
    return tuple(coords.get(k, record.get(k)) for k in by)


def group_stats(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """Group records by coords and summarize a numeric field into
    MetricTable-shaped rows. by: [coord names] ([] = one overall
    group); value: field name; stats fixed: n/median/mean/min/max +
    share_negative (useful for deltas).

    `on_missing` says what a record without the value field means:
    `error` (default) refuses by name, because a mean over the records
    that happened to have the field is the kind of number nobody
    notices is wrong; `skip` omits them and REPORTS the count, which is
    what a judged corpus needs — an unreadable verdict is not a zero
    (task 000356), and the rows that were dropped must be visible."""
    from statistics import mean, median
    recs = _items(records)
    by = params.get("by") or []
    value_field = params["value"]
    on_missing = str(params.get("on_missing", "error"))
    if on_missing not in ("error", "skip"):
        raise ValueError(
            f"group-stats on_missing must be 'error' or 'skip', not {on_missing!r}")
    groups: dict[tuple, list[float]] = {}
    n_missing = 0
    for r in recs:
        if value_field not in r or r[value_field] is None:
            if on_missing == "skip":
                n_missing += 1
                continue
            raise ValueError(
                f"group-stats: record {r.get('id')!r} has no {value_field!r} "
                f"field. Set on_missing: 'skip' if absent values are expected "
                f"(a judge that could not be read, an unscored item) — the "
                f"count is then reported on the table.")
        key = _group_key(r, by)
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
    return {"kind": "records/table",
            "name": params.get("name", f"{value_field}-stats"),
            "description": params.get("description", ""),
            "row_axis": "condition", "columns": columns, "rows": rows,
            **({"n_missing": n_missing} if n_missing else {})}


def union(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Concatenate record streams, structurally recording source
    segments; each record gains a batch coordinate named after its
    input port. Collections never mutate — growth is union.

    A union of vector collections stays a vector collection (task
    000368): a base capture and an adapted capture come from two model
    nodes, and `direction/fit` reads ONE collection whose items
    are grouped on a coordinate — the port name, on the `batch_axis`
    coordinate, is that grouping. Every item carries its own `space`, so
    the header carries no union of layers. Cross-model comparison is a
    union followed by the direction algebra, no bespoke block."""
    from mechbench_compute import shapes as S
    from mechbench_compute.lexicon import kinds as K

    batch_axis = params.get("batch_axis", "batch")
    ports = sorted(inputs.keys())

    def _as_collection(port: str, value: Any) -> Any:
        # A single kinded object on a port — a direction from
        # `direction/fit`, say — is a collection of one; it
        # takes the port's name as its id when it carries none, so the
        # union's items stay distinguishable by key.
        if isinstance(value, Mapping) and K.item_kind_of(value) is None \
                and isinstance(value.get("kind"), str):
            try:
                name, plural = K.resolve_kind(str(value["kind"]), warn=False)
            except KeyError:
                return value
            if not plural and name in K.BY_KIND:
                item = dict(value)
                if item.get("id") is None:
                    item["id"] = port
                return K.collection(name, [item])
        return value

    inputs = {p: _as_collection(p, inputs[p]) for p in ports}
    vector_inputs = [inputs[p] for p in ports]

    def _vectors(v: Any) -> bool:
        ik = K.item_kind_of(v) if isinstance(v, Mapping) else None
        return ik is not None and K.satisfies(ik, "activations/vector")

    if ports and all(_vectors(v) for v in vector_inputs):
        first = vector_inputs[0]
        rows: list[dict[str, Any]] = []
        segments = []
        for port in ports:
            rec = inputs[port]
            rec_rows = K.items_of(rec)
            segments.append({"source": port, "count": len(rec_rows)})
            for r in rec_rows:
                item = dict(r)
                item["space"] = S.space_of(r, header=rec)
                item["coords"] = {**S.coords_of(r), batch_axis: port}
                # The retired flattened spelling is not carried forward.
                for k in ("layer", "head", "label"):
                    item.pop(k, None)
                rows.append(item)
        header = {k: v for k, v in first.items()
                  if k not in ("kind", "item_kind", "key", "items", "rows",
                               "layers", "segments", "model")}
        return K.collection("activations/vector", rows, **header, segments=segments)
    segments = []
    records = []
    for port in ports:
        recs = _items(inputs[port])
        segments.append({"source": port, "count": len(recs)})
        for r in recs:
            records.append({**r, "coords": {**r.get("coords", {}),
                                            batch_axis: port}})
    # A union of ONE kind stays that kind. The items are unchanged but
    # for a coordinate, so a collection of adapter deltas unioned with
    # another is still a collection of adapter deltas — and the metrics
    # its kind declares still apply to it. Mixed inputs fall back to the
    # root, which is the only thing they have in common.
    return K.collection(_shared_item_kind(inputs, ports), records,
                        segments=segments)


def _shared_item_kind(inputs: Mapping[str, Any], ports: Sequence[str]) -> str:
    from mechbench_compute.lexicon import kinds as K

    kinds = {K.item_kind_of(inputs[p]) if isinstance(inputs[p], Mapping) else None
             for p in ports}
    if len(kinds) == 1:
        only = kinds.pop()
        if isinstance(only, str) and only in K.BY_KIND:
            return only
    return "records/record"


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
    recs = _items(records)
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
    return {"kind": "records/table",
            "name": params.get("name", "records"),
            "description": params.get("description", ""),
            "row_axis": params.get("row_axis", "record"),
            "columns": [{"name": k, "dtype": d} for k, d in dtypes.items()],
            "rows": rows}


def viz_spec(records: Any, params: Mapping[str, Any],
               source_label: str | None = None) -> dict[str, Any]:
    """A chart as a bench object: how to present an upstream table, stored
    beside it rather than drawn once and thrown away.

    When the executor knows the input's label the spec REFERENCES it
    (`source`, lineage-true, so the chart re-renders as the data
    changes); otherwise the rows ride inline (`data.rows`) and the viz
    stays self-contained. Renamed from `viz_spec` by task 000309 — "viz"
    is the level of abstraction the primitive targets.
    """
    enc = params.get("encoding") or {}
    x = enc.get("x") or params.get("x")
    y = enc.get("y") or params.get("y")
    if not x or not y:
        raise ValueError("viz/spec needs encoding.x and encoding.y")
    mark = params.get("mark", "bar")
    if mark not in ("bar", "line", "point"):
        raise ValueError(
            f"records/plot mark must be 'bar', 'line' or 'point', not {mark!r}")
    spec: dict[str, Any] = {
        "kind": "records/chart",
        "title": params.get("title", ""),
        "mark": mark,
        "encoding": {"x": x, "y": y,
                     **({"series": enc["series"]} if enc.get("series") else {})},
    }
    if source_label:
        spec["source"] = source_label
    else:
        recs = (records["rows"] if isinstance(records, Mapping)
                and isinstance(records.get("rows"), list) else _items(records))
        rows = []
        for r in recs:
            row = {k: v for k, v in r.items() if k != "coords"}
            row.update(r.get("coords", {}) if isinstance(r, Mapping) else {})
            rows.append(row)
        spec["data"] = {"rows": rows}
    return spec


_WORD_RE = None


def _words_of(text: str, lowercase: bool, min_length: int) -> list[str]:
    global _WORD_RE
    import re

    if _WORD_RE is None:
        _WORD_RE = re.compile(r"[A-Za-z'\u2019-]+")
    words = _WORD_RE.findall(text)
    if lowercase:
        words = [w.lower() for w in words]
    if min_length > 1:
        words = [w for w in words if len(w) >= min_length]
    return words


def _vocabulary_of(name: str, items: Any) -> list[str]:
    """A list measure's vocabulary: a list of outcomes, or the outcomes of
    a target map (`weights` keys, or `uniform`). Pure, so no transform: a
    rung's narrower vocabulary is listed, and an outcome outside it but in
    the full map is still a real outcome, not an unknown one."""
    if isinstance(items, (list, tuple)):
        return [str(x) for x in items]
    if isinstance(items, Mapping):
        if items.get("transform"):
            raise ValueError(
                f"text/measure measure {name!r}: `items` takes a list or an "
                "untransformed map; list the outcomes to narrow it")
        if isinstance(items.get("weights"), Mapping):
            return [str(k) for k in items["weights"]]
        if isinstance(items.get("uniform"), (list, tuple)):
            return [str(x) for x in items["uniform"]]
    raise ValueError(
        f"text/measure measure {name!r}: `items` is a list of outcomes or a "
        "map with `weights` or `uniform`")


def text_stats(inputs: Mapping[str, Any],
               params: Mapping[str, Any]) -> Any:
    """text/measure — configurable per-text measurements
    over a corpus of records or a document_collection (the generate
    block's output). The measurement layer the story-corpus readouts
    need (meta-leak counts, opening-phrase counts, lexical spread,
    corpus-frequency of vocabulary) with lineage instead of ad-hoc
    scripts.

    params:
      field       which field holds the text (default "text")
      measures    list of measure specs, each {"kind", "name", ...}:
        {"type": "pattern", "name": n, "patterns": [regex...],
         "where": "prefix" | "anywhere" (default), "ignore_case": bool}
            → per-record 0/1: does ANY pattern match?
        {"type": "lexical", "name": n, "lowercase": bool=true,
         "min_length": int=1}
            → per-record word/distinct-word counts and duplication
              (1 − distinct/total).
        {"type": "corpus_frequency", "name": n, "stat":
         "mean_log10" (default) | "mean" | "coverage",
         "lowercase": bool=true, "min_length": int=1,
         "frequencies": {word: count}   — or wire a `frequencies`
         input whose payload carries {"weights": {...}}}
            → per-record statistic over the reference frequency of the
              words that appear in it ("coverage" = fraction of words
              found in the table). The lexical-novelty instrument:
              rarer vocabulary ⇒ lower mean_log10.
        {"type": "list", "name": n, "separator": ", ", "extract": regex,
         "items": [...] | target spec, "count": int, "ignore_case": bool}
            → per-record parse of a drawn list (000548): items,
              distinct, duplicates, unknown (outside `items`), first,
              parsed, valid. `extract`'s first group (else its whole
              match) is the list; without it the whole text is.
      mode        "annotate" (default): records with measure fields
                  added — feed select/group-stats/table downstream.
                  "corpus": ONE summary record — pattern counts and
                  rates, corpus-wide distinct words and duplication,
                  means of per-record frequency stats.
                  "items": one record per distinct item a `list` measure
                  parsed, with its count — what the corpus SAID, whether
                  or not `items` contains it (000551).
    """
    import math as _math
    import re

    raw = inputs.get("records")
    if raw is None:
        raw = inputs.get("documents")
    if raw is None:
        raise ValueError(
            "text/measure needs texts on its `records` or `documents` port")
    recs = _items(raw)
    field = "text"
    measures = params.get("measures") or []
    mode = params.get("mode", "annotate")
    if mode not in ("annotate", "corpus", "items"):
        raise ValueError(
            "text/measure mode must be 'annotate', 'corpus' or 'items', not "
            f"{mode!r}")
    # `keep` (task 000368): an annotated row carries the whole item —
    # text, trace, metadata — not just id + coords + measures, so a
    # capture downstream can replay the story it was labelled on.
    keep = bool(params.get("keep", False))

    freq_input = inputs.get("frequencies")
    if isinstance(freq_input, Mapping):
        freq_input = (freq_input.get("weights")
                      or freq_input.get("frequencies") or freq_input)

    def coords_of(r):
        if isinstance(r.get("coords"), Mapping):
            return dict(r["coords"])
        meta = r.get("metadata")
        if isinstance(meta, Mapping) and isinstance(meta.get("coords"),
                                                    Mapping):
            return dict(meta["coords"])
        return {}

    compiled = []
    for m in measures:
        kind = m.get("type") or m.get("kind")
        name = m.get("name") or kind
        if kind not in ("pattern", "lexical", "corpus_frequency", "list"):
            raise ValueError(
                f"text/measure: unknown measure type {kind!r}: one of "
                "'pattern', 'lexical', 'corpus_frequency', 'list'")
        if kind == "pattern":
            flags = re.IGNORECASE if m.get("ignore_case") else 0
            pats = [re.compile(pat, flags) for pat in m["patterns"]]
            where = m.get("where", "anywhere")
            if where not in ("anywhere", "prefix"):
                raise ValueError(
                    f"text/measure measure {name!r}: where must be 'anywhere' "
                    f"or 'prefix', not {where!r}")
            compiled.append((name, kind, {"patterns": pats,
                                          "where": where}))
        elif kind == "lexical":
            compiled.append((name, kind,
                             {"lowercase": bool(m.get("lowercase", True)),
                              "min_length": int(m.get("min_length", 1))}))
        elif kind == "corpus_frequency":
            table = m.get("frequencies") or freq_input
            if not isinstance(table, Mapping) or not table:
                raise ValueError(
                    f"text/measure measure {name!r}: no frequency table "
                    "(inline `frequencies` or a wired frequencies input)")
            lower = bool(m.get("lowercase", True))
            tbl = {str(k).lower() if lower else str(k): float(v)
                   for k, v in table.items()}
            stat = m.get("stat", "mean_log10")
            if stat not in ("mean_log10", "mean", "coverage"):
                raise ValueError(
                    f"text/measure measure {name!r}: stat must be 'mean_log10', "
                    f"'mean' or 'coverage', not {stat!r}")
            compiled.append((name, kind,
                             {"table": tbl,
                              "stat": stat,
                              "lowercase": lower,
                              "min_length": int(m.get("min_length", 1))}))
        elif kind == "list":
            fold = bool(m.get("ignore_case", False))
            vocab = m.get("items")
            separator = str(m.get("separator", ", "))
            if separator == "":
                raise ValueError(f"text/measure measure {name!r}: a list needs a separator")
            compiled.append((name, kind, {
                "separator": separator,
                "extract": re.compile(m["extract"], re.DOTALL) if m.get("extract") else None,
                "vocab": ({(k.casefold() if fold else k) for k in _vocabulary_of(name, vocab)}
                          if vocab is not None else None),
                "count": int(m["count"]) if m.get("count") is not None else None,
                "fold": fold}))
        else:
            raise ValueError(f"text/measure: unknown measure kind {kind!r}")

    out = []
    corpus_items: dict[str, set[str]] = {}
    # What the corpus said, item by item (000551): the vocabulary labels
    # an answer, it does not decide whether the answer counts.
    said: dict[str, dict[str, dict]] = {}
    corpus_words: list[str] = []
    for r in recs:
        text = str(r.get(field, ""))
        row = {**(r if keep else {}), "id": r.get("id"), "coords": coords_of(r)}
        for name, kind, cfg in compiled:
            if kind == "pattern":
                probe = text.lstrip() if cfg["where"] == "prefix" else text
                hit = any((p.match(probe) if cfg["where"] == "prefix"
                           else p.search(probe)) for p in cfg["patterns"])
                row[name] = 1 if hit else 0
            elif kind == "lexical":
                words = _words_of(text, cfg["lowercase"], cfg["min_length"])
                distinct = len(set(words))
                row[f"{name}_words"] = len(words)
                row[f"{name}_distinct"] = distinct
                row[f"{name}_dup"] = (round(1.0 - distinct / len(words), 4)
                                      if words else 0.0)
                corpus_words.extend(words)
            elif kind == "list":
                body: str | None = text
                if cfg["extract"] is not None:
                    found = cfg["extract"].search(text)
                    body = (found.group(1) if found and found.groups() else
                            found.group(0) if found else None)
                items = ([i.strip() for i in body.split(cfg["separator"])]
                         if body is not None else [])
                items = [i for i in items if i]
                keys = [i.casefold() if cfg["fold"] else i for i in items]
                distinct = len(set(keys))
                unknown = (sum(1 for k in keys if k not in cfg["vocab"])
                           if cfg["vocab"] is not None else None)
                row[f"{name}_parsed"] = 1 if body is not None else 0
                row[f"{name}_items"] = len(items)
                row[f"{name}_distinct"] = distinct
                row[f"{name}_duplicates"] = len(items) - distinct
                if unknown is not None:
                    row[f"{name}_unknown"] = unknown
                row[f"{name}_first"] = items[0] if items else None
                row[f"{name}_valid"] = int(
                    body is not None and bool(items)
                    and distinct == len(items) and not unknown
                    and (cfg["count"] is None or len(items) == cfg["count"]))
                corpus_items.setdefault(name, set()).update(keys)
                for pos, (item, key) in enumerate(zip(items, keys)):
                    tally = said.setdefault(name, {}).setdefault(key, {
                        "item": item, "count": 0, "lists": 0, "first": 0,
                        "in_vocabulary": None if cfg["vocab"] is None else key in cfg["vocab"]})
                    tally["count"] += 1
                    tally["first"] += int(pos == 0)
                    if key not in keys[:pos]:
                        tally["lists"] += 1
            else:  # corpus_frequency
                words = _words_of(text, cfg["lowercase"], cfg["min_length"])
                vals = [cfg["table"][w] for w in words if w in cfg["table"]]
                cov = (len(vals) / len(words)) if words else 0.0
                if cfg["stat"] == "coverage":
                    row[name] = round(cov, 4)
                elif cfg["stat"] == "mean":
                    row[name] = (round(sum(vals) / len(vals), 4)
                                 if vals else None)
                else:  # mean_log10
                    row[name] = (round(sum(_math.log10(v) for v in vals)
                                       / len(vals), 4) if vals else None)
                row[f"{name}_coverage"] = round(cov, 4)
        out.append(row)

    if mode == "annotate":
        return out

    if mode == "items":
        if not any(kind == "list" for _, kind, _ in compiled):
            raise ValueError(
                "text/measure mode 'items' needs a `list` measure: it tallies "
                "what the lists said")
        rows = []
        for name, tally in said.items():
            total = sum(t["count"] for t in tally.values()) or 1
            for t in sorted(tally.values(), key=lambda t: (-t["count"], t["item"])):
                rows.append({"id": f"{name}:{t['item']}",
                             "coords": {"measure": name, "item": t["item"]},
                             **t, "share": round(t["count"] / total, 6)})
        return rows

    summary: dict[str, Any] = {"id": "corpus", "coords": {},
                               "n_texts": len(out)}
    for name, kind, cfg in compiled:
        if kind == "pattern":
            n = sum(r[name] for r in out)
            summary[f"{name}_count"] = n
            summary[f"{name}_rate"] = (round(n / len(out), 4)
                                       if out else 0.0)
        elif kind == "list":
            n_lists = len(out) or 1
            n_items = sum(r[f"{name}_items"] for r in out)
            summary[f"{name}_parsed_rate"] = round(sum(r[f"{name}_parsed"] for r in out) / n_lists, 4)
            summary[f"{name}_mean_items"] = round(n_items / n_lists, 4)
            summary[f"{name}_duplicate_rate"] = round(
                sum(1 for r in out if r[f"{name}_duplicates"] > 0) / n_lists, 4)
            summary[f"{name}_valid_rate"] = round(sum(r[f"{name}_valid"] for r in out) / n_lists, 4)
            summary[f"{name}_distinct_items"] = len(corpus_items.get(name, set()))
            if cfg["vocab"] is not None:
                summary[f"{name}_unknown_rate"] = (
                    round(sum(r[f"{name}_unknown"] for r in out) / n_items, 4)
                    if n_items else 0.0)
        elif kind == "lexical":
            summary[f"{name}_corpus_words"] = len(corpus_words)
            summary[f"{name}_corpus_distinct"] = len(set(corpus_words))
            summary[f"{name}_corpus_dup"] = (
                round(1.0 - len(set(corpus_words)) / len(corpus_words), 4)
                if corpus_words else 0.0)
        else:
            vals = [r[name] for r in out if r.get(name) is not None]
            summary[f"{name}_mean"] = (round(sum(vals) / len(vals), 4)
                                       if vals else None)
    return [summary]


PURE_BLOCKS: dict[str, Callable[..., Any]] = {
    # Keyed by the one current name. `grid` and `factor-cross`, the
    # spellings this op had before, are refused since 0.82.0 —
    # `lexicon.explain_unknown` names the replacement.
    "records/cross":
        lambda inputs, params: _coll(factor_cross(params)),
    "records/fill":
        lambda inputs, params: _coll(template(_items(inputs["records"]), params)),
    "records/rename":
        lambda inputs, params: _coll(rename(inputs["records"], params)),
    "records/select":
        lambda inputs, params: _selected(inputs["records"], params),
    "records/subtract":
        lambda inputs, params: _coll(paired_delta(inputs["records"], params)),
    "records/summarize":
        lambda inputs, params: group_stats(inputs["records"], params),
    "records/tabulate":
        lambda inputs, params: table_from_records(inputs["records"], params),
    "records/union":
        lambda inputs, params: union(inputs, params),
    "records/zip":
        lambda inputs, params: zip_branches(inputs, params),
    "text/measure":
        lambda inputs, params: _coll(text_stats(inputs, params)),
    "eval/expect":
        lambda inputs, params: eval_expectation(inputs, params),
    # Interp readouts (the mechbench-experiments port): pure numpy over
    # residual_vectors records — no model, no weights.
    "geometry/compare":
        lambda inputs, params: _geometry_similarity(inputs, params),
    # Weight space (task 000458): an adapter's own low-rank factors,
    # read with no model and no forward pass.
    "adapter/measure":
        lambda inputs, params: _measure_adapter(inputs["adapter"], params),
}


def zip_branches(inputs: Mapping[str, Any],
                 params: Mapping[str, Any]) -> dict[str, Any]:
    """`records/zip` (task 000398): align several branches' records into
    one record per key, keeping which branch each came from.

    Two branches over the same prompts produce two streams; a judge that
    compares them needs record 7 of one beside record 7 of the other.
    `records/union` concatenates — the items stay separate, distinguished
    by a coordinate. This pairs them.

    The key is `id` by default, or a list of coordinate names (`by:
    ["prompt", "seed"]`), which is what to use when two branches number
    their records differently but share a design.

    A branch is named by the node it came from, unless `names` says
    otherwise — the names become the keys of each output record's
    `branches` map, and the field prefixes under `flatten`.
    """
    from mechbench_compute.lexicon import kinds as K

    edges = inputs.get("branches") or []
    if isinstance(edges, Mapping):            # one branch, given inline
        edges = [{"node": "branch", "value": edges}]
    if len(edges) < 2:
        raise ValueError(
            "records/zip needs at least two branches: wire one edge per "
            "branch onto its `branches` port. With one it would be a copy.")
    names = list(params.get("names") or [])
    if names and len(names) != len(edges):
        raise ValueError(
            f"`names` has {len(names)} entries for {len(edges)} branches")
    labels = names or [str(e.get("node") or i) for i, e in enumerate(edges)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two branches share a name: {labels}")

    by = params.get("by", "id")
    on_mismatch = str(params.get("on_mismatch", "fail"))
    if on_mismatch not in ("fail", "drop", "placeholder"):
        raise ValueError(
            f"on_mismatch is 'fail', 'drop' or 'placeholder', not {on_mismatch!r}")

    def key_of(rec: Mapping[str, Any]) -> Any:
        if by == "id":
            return str(rec.get("id"))
        coords = rec.get("coords") or {}
        missing = [c for c in by if c not in coords]
        if missing:
            raise ValueError(
                f"record {rec.get('id')!r} has no {', '.join(missing)} "
                f"coordinate to zip by")
        return tuple(str(coords[c]) for c in by)

    indexed: list[dict[Any, Mapping[str, Any]]] = []
    for e in edges:
        rows: dict[Any, Mapping[str, Any]] = {}
        for rec in _items(e.get("value")):
            k = key_of(rec)
            if k in rows:
                raise ValueError(
                    f"branch {e.get('node')!r} has two records keyed {k!r}; "
                    f"zip needs one per key (try `by` on more coordinates)")
            rows[k] = rec
        indexed.append(rows)

    everywhere = set(indexed[0])
    anywhere = set(indexed[0])
    for rows in indexed[1:]:
        everywhere &= set(rows)
        anywhere |= set(rows)
    missing = anywhere - everywhere
    if missing and on_mismatch == "fail":
        sample = ", ".join(repr(k) for k in sorted(missing, key=str)[:4])
        raise ValueError(
            f"{len(missing)} key(s) are not in every branch ({sample}"
            f"{'…' if len(missing) > 4 else ''}). `on_mismatch: \"drop\"` "
            f"keeps the keys every branch has; `\"placeholder\"` keeps them "
            f"all and marks what is absent.")
    keys = sorted(everywhere if on_mismatch != "placeholder" else anywhere,
                  key=str)

    flatten = bool(params.get("flatten", False))
    out: list[dict[str, Any]] = []
    for k in keys:
        present = [rows.get(k) for rows in indexed]
        first = next(r for r in present if r is not None)
        rec: dict[str, Any] = {
            "id": str(first.get("id")) if by == "id" else "-".join(k),
            "coords": dict(first.get("coords") or {}),
        }
        if flatten:
            for label, r in zip(labels, present):
                for field, value in (r or {}).items():
                    if field not in ("id", "coords", "kind"):
                        rec[f"{label}_{field}"] = value
                if r is None:
                    rec[f"{label}_missing"] = True
        else:
            rec["branches"] = {
                label: (dict(r) if r is not None else {"missing": True})
                for label, r in zip(labels, present)}
        out.append(rec)

    return K.collection(
        "records/record", out,
        branches=[{"name": label, "source": e.get("node"), "n": len(rows)}
                  for label, e, rows in zip(labels, edges, indexed)],
        zipped={"by": by, "keys": len(out), "on_mismatch": on_mismatch,
                "dropped": sorted(map(str, missing))[:50] if missing else []},
        name=params.get("name"),
        description=params.get("description"),
    )


def _selected(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """A filter keeps the kind; a projection does not. Selecting some of
    a collection's items leaves each item exactly as it was, so a subset
    of adapter deltas is still adapter deltas and still compares by what
    that kind declares. `fields` rewrites the items, and what is left
    may no longer satisfy the kind — that lands as a plain record."""
    from mechbench_compute.lexicon import kinds as K

    items = select(records, params)
    if params.get("fields"):
        return _coll(items)
    kind = K.item_kind_of(records) if isinstance(records, Mapping) else None
    return K.collection(kind if kind in K.BY_KIND else "records/record", items)


def _measure_adapter(adapter: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    from mechbench_compute.weights import measure_adapter

    payload = adapter.get("payload", adapter) if isinstance(adapter, Mapping) else adapter
    return measure_adapter(payload, params)

# Trajectory readouts (task 000368): pure numpy over trajectory records.
from mechbench_compute.trajectory import PURE as _TRAJECTORY_PURE

PURE_BLOCKS.update(_TRAJECTORY_PURE)


def _geometry_similarity(inputs, params):
    from mechbench_compute.similarity import geometry_similarity

    return geometry_similarity(inputs, params)


def eval_expectation(inputs: Mapping[str, Any],
                     params: Mapping[str, Any]) -> dict[str, Any]:
    """The first member of the eval block family (~canonical/ops/eval/):
    judge decision-read results against per-condition EXPECTATIONS
    carried as data, publishing a metric table with verdicts.

    Expectation kinds (per record, joined on id):
      {"type": "uniform", "over": [outcomes], "max_kl_bits": t}
          -> kl_bits from uniform over the outcome masses; pass iff
             kl_bits <= t and the outcomes carry real mass.
      {"type": "answer", "value": tok, "min_p": t}
          -> p_expected from the read's top tokens; pass iff >= t.
      {"type": "min_entropy", "bits": t}
          -> pass iff the decision entropy >= t (diversity floor).
      {"type": "weights", "weights": {outcome: w}, "max_kl_bits": t}
          -> kl_bits from the NORMALIZED weights over the outcome
             masses (the shaped-target battery: a rung is judged
             against its OWN target, not uniform); pass iff <= t.
      {"type": "absent", "over": [outcomes], "max_p": t}
          -> mass, the total on outcomes that should not be said (a
             list slot's repeats); pass iff mass <= t. Every named
             outcome must have been READ (in `tracked`): absence is
             never inferred from an outcome the read did not score.

    The aggregate row (id "ALL") carries the pass rate — the number a
    publication cites.
    """
    import math

    results = _items(inputs["results"])
    expectations = {r["id"]: r["expect"] for r in _items(inputs["expectations"])}
    from mechbench_compute import shapes as S
    from mechbench_compute.lexicon import kinds as K

    rows = []
    n_pass = 0
    n_judged = 0
    n_unjudgeable = 0
    for c in results:
        exp = expectations.get(c["id"])
        if not exp:
            continue
        expect_type = exp.get("type") or exp["kind"]
        # One shape to read: the distribution's `tracked` holds each
        # named outcome's mass, `top` the ranked tokens. A read written
        # before the shape existed is read through `distribution_of`.
        dist = S.distribution_of(c)
        tracked = dist.get("tracked") or {}
        row: dict[str, Any] = {"id": c["id"], "coords": dict(c.get("coords") or {}),
                               "expect": expect_type,
                               "entropy_bits": dist.get("entropy_bits")}

        def mass_of(name: str) -> float:
            t = tracked.get(str(name))
            if t is not None and t.get("p") is not None:
                return float(t["p"])
            # Not tracked: the ranked tokens, by exact text.
            return sum(float(t["p"]) for t in dist.get("top") or []
                       if str(t["token"].get("text")).strip() == str(name).strip()
                       and t.get("p") is not None)

        ok = False
        if expect_type == "uniform":
            over = exp["over"]
            ps = [mass_of(o) for o in over]
            tot = sum(ps)
            if tot > 0:
                kl = sum(q / tot * math.log2((q / tot) / (1.0 / len(over)))
                         for q in ps if q > 0)
                row["kl_bits"] = round(kl, 4)
                row["mass"] = round(tot, 4)
                ok = kl <= float(exp.get("max_kl_bits", 0.1))
            else:
                # Nothing to judge is not a failure — it is a hole in
                # the read, and it must not masquerade as one more
                # False among real verdicts.
                row["pass"] = None
                row["note"] = "unjudgeable: no mass on any outcome in the read"
                n_unjudgeable += 1
                rows.append(row)
                continue
        elif expect_type == "weights":
            wsum = sum(float(v) for v in exp["weights"].values())
            target = {str(k): float(v) / wsum
                      for k, v in exp["weights"].items() if float(v) > 0}
            ps = [mass_of(o) for o in target]
            tot = sum(ps)
            if tot > 0:
                kl = sum(
                    (q / tot) * math.log2((q / tot) / target[o])
                    for o, q in zip(target, ps) if q > 0)
                row["kl_bits"] = round(kl, 4)
                row["mass"] = round(tot, 4)
                ok = kl <= float(exp.get("max_kl_bits", 0.1))
            else:
                row["pass"] = None
                row["note"] = "unjudgeable: no mass on any outcome in the read"
                n_unjudgeable += 1
                rows.append(row)
                continue
        elif expect_type == "answer":
            want = str(exp["value"])
            p = mass_of(want) if (want in tracked or dist.get("top")) else None
            if p == 0.0 and want not in tracked:
                p = None
            row["p_expected"] = round(p, 4) if p is not None else None
            ok = p is not None and p >= float(exp.get("min_p", 0.99))
        elif expect_type == "min_entropy":
            ok = float(dist.get("entropy_bits") or 0.0) >= float(exp["bits"])
        elif expect_type == "absent":
            over = [str(o) for o in exp["over"]]
            unread = [o for o in over if o not in tracked]
            if unread:
                row["pass"] = None
                row["note"] = f"unjudgeable: not read — {unread[:5]}"
                n_unjudgeable += 1
                rows.append(row)
                continue
            mass = sum(mass_of(o) for o in over)
            row["mass"] = round(mass, 6)
            ok = mass <= float(exp.get("max_p", 0.01))
        else:
            raise ValueError(f"unknown expectation type: {expect_type!r}")
        row["pass"] = ok
        n_judged += 1
        n_pass += int(ok)
        rows.append(row)
    return K.collection(
        "eval/verdict", rows,
        name=params.get("name", "expectation-eval"),
        description=params.get("description", ""),
        summary={"pass_rate": round(n_pass / n_judged, 4) if n_judged else None,
                 "n_pass": n_pass, "n_judged": n_judged,
                 "n_unjudgeable": n_unjudgeable})


# Directions as first-class objects (task 000367): pure producers and
# arithmetic live in `directions.py`; the vocabulary projection is a model
# block in the executor.
from mechbench_compute.directions import (
    PURE_DIRECTION_BLOCKS as _DIRECTION_BLOCKS,
)

PURE_BLOCKS.update(_DIRECTION_BLOCKS)

# Exact generic monoid reduces (task 000406): sum, top-k, histogram.
from mechbench_compute.reduce import PURE_REDUCE_BLOCKS as _REDUCE_BLOCKS

PURE_BLOCKS.update(_REDUCE_BLOCKS)

# Tool handlers are ordinary blocks (task 000340): what a model may
# call is what the platform can already do.
from mechbench_compute.tools import PURE_TOOL_BLOCKS as _TOOL_BLOCKS

PURE_BLOCKS.update(_TOOL_BLOCKS)

# Variety as tree structure (task 000430): an MST over pairwise
# distance, whose edge statistics separate a collapsed corpus from a
# clustered one from an evenly varied one.
from mechbench_compute.trees import PURE_TREE_BLOCKS as _TREE_BLOCKS

PURE_BLOCKS.update(_TREE_BLOCKS)
