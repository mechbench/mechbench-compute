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

import math

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _transcript_mod():
    from mechbench_compute import transcript
    return transcript


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


def arch_header(arch: Any) -> dict[str, Any]:
    """The model's depth landmarks, as a result header carries them: how
    many layers, which attend globally, and where fresh keys and values
    stop. A figure that carries these draws them on its depth axis, so a
    layer is the same place in every figure (VISUALIZATION.md).

    A model family with no hybrid attention or key/value sharing has no
    such landmarks, and says so by leaving them out rather than
    inventing an empty list."""
    out: dict[str, Any] = {"n_layers": int(arch.n_layers)}
    globals_ = getattr(arch, "global_layers", None)
    if globals_ is not None:
        out["global_layers"] = [int(i) for i in globals_]
    first_shared = getattr(arch, "first_kv_shared_layer", None)
    if first_shared is not None:
        out["first_kv_shared_layer"] = int(first_shared)
    return out


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
        if kind not in ("pattern", "lexical", "corpus_frequency", "list", "capture"):
            raise ValueError(
                f"text/measure: unknown measure type {kind!r}: one of "
                "'pattern', 'lexical', 'corpus_frequency', 'list', 'capture'")
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
        elif kind == "capture":
            # One value, under the name asked for (000618). A `list`
            # measure could already read it, as "the first item of a
            # list this text said" — with five columns of list
            # statistics and a `_first` suffix nobody wanted.
            if not m.get("pattern"):
                raise ValueError(
                    f"text/measure measure {name!r}: a capture needs a `pattern`")
            take = m.get("take", "first")
            if take not in ("first", "last"):
                raise ValueError(
                    f"text/measure measure {name!r}: take is 'first' or "
                    f"'last', not {take!r}")
            as_ = m.get("as", "string")
            if as_ not in ("string", "number"):
                raise ValueError(
                    f"text/measure measure {name!r}: `as` is 'string' or "
                    f"'number', not {as_!r}")
            on_missing = m.get("on_missing", "null")
            if on_missing not in ("null", "error"):
                raise ValueError(
                    f"text/measure measure {name!r}: on_missing is 'null' or "
                    f"'error', not {on_missing!r}")
            fold = bool(m.get("ignore_case", False))
            vocab = m.get("items")
            compiled.append((name, kind, {
                "pattern": re.compile(
                    m["pattern"], re.IGNORECASE | re.DOTALL if fold else re.DOTALL),
                "group": int(m.get("group", 1)),
                # A vocabulary both constrains and CANONICALISES: a text
                # that says "Ana" captures the `ana` the vocabulary
                # spells, which is what the value is compared against
                # downstream. A match outside it is no match.
                "vocab": ({(k.casefold() if fold else k): k
                           for k in _vocabulary_of(name, vocab)}
                          if vocab is not None else None),
                "fold": fold,
                "take": take, "as": as_, "on_missing": on_missing}))
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
    captured: dict[str, list[Any]] = {}
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
            elif kind == "capture":
                found = list(cfg["pattern"].finditer(text))
                hit = (found[-1] if cfg["take"] == "last" else found[0]) if found else None
                value: Any = None
                if hit is not None:
                    group = cfg["group"] if hit.groups() else 0
                    try:
                        value = hit.group(group)
                    except (IndexError, re.error):
                        raise ValueError(
                            f"text/measure measure {name!r}: the pattern has no "
                            f"group {cfg['group']}") from None
                if value is not None and cfg["vocab"] is not None:
                    value = cfg["vocab"].get(
                        str(value).casefold() if cfg["fold"] else str(value))
                if value is None and cfg["on_missing"] == "error":
                    raise ValueError(
                        f"text/measure measure {name!r}: {r.get('id')!r} says "
                        "nothing the pattern matches. Set on_missing: 'null' if "
                        "that is expected.")
                if value is not None and cfg["as"] == "number":
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(
                            f"text/measure measure {name!r}: {r.get('id')!r} "
                            f"captured {value!r}, which is not a number") from None
                    if value.is_integer():
                        value = int(value)
                if value is not None:
                    row[name] = value
                    captured.setdefault(name, []).append(value)
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
        elif kind == "capture":
            values = captured.get(name, [])
            summary[f"{name}_captured"] = len(values)
            summary[f"{name}_rate"] = round(len(values) / len(out), 4) if out else 0.0
            numbers = [v for v in values if isinstance(v, (int, float))]
            if numbers:
                summary[f"{name}_mean"] = round(sum(numbers) / len(numbers), 4)
            else:
                # What the texts said, commonest first — the tally a
                # captured label is usually wanted for.
                counts: dict[str, int] = {}
                for v in values:
                    counts[str(v)] = counts.get(str(v), 0) + 1
                summary[f"{name}_values"] = [
                    {"value": v, "count": n} for v, n in
                    sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
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
    "text/render":
        lambda inputs, params: _transcript_mod().render_records(inputs, params),
    "text/extend":
        lambda inputs, params: _transcript_mod().extend(inputs, params),
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
from mechbench_compute.blocks.cell_rows import cell_rows  # noqa: F401
from mechbench_compute.blocks.grid_rows import grid_rows  # noqa: F401
from mechbench_compute.blocks.group_key import _group_key  # noqa: F401
from mechbench_compute.blocks.interval_of import _interval_of  # noqa: F401
from mechbench_compute.blocks.items import _items  # noqa: F401
from mechbench_compute.blocks.coll import _coll  # noqa: F401

PURE_BLOCKS.update(_TREE_BLOCKS)

# An operation that has its own file (docs/OPS_LAYOUT.md) and runs with
# no executor is callable here by name, as the ones above are. It is
# looked up when asked for, never at import: finding the operations
# imports their files, their files import helpers from packages like
# this one, and a package's `__init__` that went looking for operations
# while one of them was half-imported would find it without its OP.
class _PureBlocks(dict):
    @staticmethod
    def _elsewhere() -> frozenset[str]:
        from mechbench_compute import ops

        return ops.standalone()

    def __contains__(self, name: object) -> bool:
        return dict.__contains__(self, name) or name in self._elsewhere()

    def __missing__(self, name: str):
        if name not in self._elsewhere():
            raise KeyError(name)
        from mechbench_compute import ops

        return lambda inputs, params: ops.run_standalone(name, inputs, params)

    def __iter__(self):
        yield from dict.__iter__(self)
        yield from sorted(n for n in self._elsewhere() if not dict.__contains__(self, n))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def keys(self):
        return list(self)

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default


PURE_BLOCKS = _PureBlocks(PURE_BLOCKS)
