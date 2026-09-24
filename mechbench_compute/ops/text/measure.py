from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="text/measure",
    summary=(
        "Measure each text in a corpus — pattern hits, word and distinct-word "
        "counts, vocabulary rarity against a frequency table — and either "
        "annotate the records or summarise the corpus."
    ),
    description="""\
Each entry of `measures` is applied to every record's `text`:

| `type` | Fields written per record | Options |
|---|---|---|
| `pattern` | `<name>`: 1 if any regex matches, else 0 | `patterns` (list of regexes), `where`: `"anywhere"` or `"prefix"` (must match at the start), `ignore_case` |
| `lexical` | `<name>_words`, `<name>_distinct`, `<name>_dup` (1 − distinct/words) | `lowercase` (default true), `min_length`, `exclude` (words not counted) |
| `corpus_frequency` | `<name>`: the statistic over the reference frequency of the text's words; `<name>_coverage`: the fraction of words found in the table | `frequencies` (word → count, or wire a `frequencies` input), `stat`: `"mean_log10"` (rarer vocabulary ⇒ lower), `"mean"` or `"coverage"`, `lowercase`, `min_length` |
| `list` | `<name>_parsed` (1 if the list was found), `<name>_items`, `<name>_distinct`, `<name>_duplicates`, `<name>_unknown` (items outside `items`, when given), `<name>_first`, `<name>_valid` (found, no duplicates, nothing unknown, and `count` items when given) | `separator` (default `", "`), `extract` (a regex whose first group is the list; the whole text without it), `items` (the vocabulary: a list, or a map's `weights` or `uniform`), `count`, `ignore_case` |
| `capture` | `<name>`: the value the pattern's group held, absent when nothing matched | `pattern` (the regex), `group` (default 1; the whole match when the pattern has none), `as`: `"string"` or `"number"`, `take`: the `"first"` match or the `"last"`, `on_missing`: `"null"` or `"error"`, `items` (a vocabulary, which canonicalises the value), `ignore_case` |

A **capture** is the one that reads a value out rather than counting or
tallying: a rating the model wrote, a label it chose, a field of the JSON
it produced — and a local model is the case that needs it, since
`json_mode` is refused there and the reply is only ever text. The value
lands under the name given, as a number when asked, so it can be bound
into a `records/map`, grouped by `records/summarize` or plotted without a
rename. (A `list` measure of one thing could always read it; it wrote
`<name>_first` and five columns of list statistics to do so.)

In `annotate` mode the output is the records with those fields added —
ready for `records/select`, `records/summarize` or `trajectory/capture`. To
group on a measure downstream, `records/rename` it into `coords`. In
`corpus` mode it is one summary record: per pattern a count and rate,
corpus-wide word and distinct-word counts and duplication, and the mean of
each frequency statistic; per list, the parsed, duplicate and valid rates,
mean items, distinct items across the corpus, and the unknown-item rate.

In `items` mode it is one record per distinct item the `list` measures
parsed — `item`, `count`, `lists` (how many lists held it), `first` (how
often it led one), `share`, and `in_vocabulary` when `items` was given.
This is what the corpus SAID, in its own vocabulary: an answer outside
the map is a row like any other, labelled rather than dropped, so "Sci-Fi"
and "Steampunk Fantasy" are readable beside the names the map has.

A `lexical` measure in `items` mode gives one record per distinct word —
`item`, `count` (occurrences), `texts` (how many texts use it: its
document frequency), and `share` (of all words) — so "97 of 100 stories
say *last*" is a row, and `records/rank` on `texts` lists the words a
corpus converges on. A word is a run of letters, apostrophes and
hyphens. `exclude` names words to leave out — the function words a
ranking would otherwise lead with — and they are left out of the
per-text counts as well. The list is the author's: no words are
excluded unless named.
""",
    inputs=(
        In("records", "records/record",
           "The texts, each in its `text` field. One of `records` and "
           "`documents` is required.", many=True, required=False),
        In("documents", "text/document",
           "A document collection, usually from `text/generate` or `text/chat`.",
           many=True, required=False),
        In("frequencies", "text/word-list",
           "A word-frequency table (`weights`: word → count) for "
           "`corpus_frequency` measures that name none of their own.",
           required=False),
    ),
    output=(
        Output('records/record', collection=True, doc='In `annotate` mode, one record per item (`id`, `coords`, the measure fields, and the whole item when `keep` is set). In `corpus` mode, a single record with the corpus summary.')
    ),
    params=(
        P("measures", "list[object]",
          "The measurements to make, each `{type, name, …}` as in the "
          "table above.",
          None, fields=(
              P("type", "string", "Which measurement. One of `type` or `kind` is required.", None,
                choices=("pattern", "lexical", "corpus_frequency", "list", "capture")),
              P("kind", "string", "The older spelling of `type`; read when `type` is absent.", None,
                choices=("pattern", "lexical", "corpus_frequency", "list", "capture")),
              P("name", "string", "The field it writes; the type by default.", None),
              P("patterns", "list[string]", "For `pattern`: the regular expressions.", None),
              P("pattern", "string",
                "For `capture`: the regular expression whose group is the value.", None),
              P("group", "int",
                "For `capture`: which group of the pattern to take; the whole match when "
                "the pattern has none.", 1),
              P("as", "string",
                "For `capture`: the type the value takes. A number that arrives as a string "
                "reads fine in a table and then fails a summary.", "string",
                choices=("string", "number")),
              P("on_missing", "string",
                "For `capture`: what a text that matches nothing means — the field is simply "
                "absent, or the run stops and names the record.", "null",
                choices=("null", "error")),
              P("where", "string", "For `pattern`: match anywhere, or only at the start.", "anywhere",
                choices=("anywhere", "prefix")),
              P("take", "string",
                "For `capture`: which match to take when the text holds several. A hand-off "
                "is the last, a header the first.", "first",
                choices=("first", "last")),
              P("ignore_case", "bool", "For `pattern` and `list`: match without regard to case.", False),
              P("lowercase", "bool", "For `lexical` and `corpus_frequency`: lowercase words first.", True),
              P("min_length", "int", "For `lexical` and `corpus_frequency`: the shortest word counted.", 1),
              P("exclude", "list[string]",
                "For `lexical`: words not counted, compared after lowercasing when "
                "`lowercase` is set. A stored word list may be given by reference.",
                None, stored="text/word-list"),
              P("frequencies", "map[string, float]",
                "For `corpus_frequency`: word → count, unless a `frequencies` input is wired. "
                "A stored word list may be given by reference.", None, stored="text/word-list"),
              P("stat", "string", "For `corpus_frequency`: the statistic.", "mean_log10",
                choices=("mean_log10", "mean", "coverage")),
              P("separator", "string", "For `list`: the text between items.", ", "),
              P("extract", "string",
                "For `list`: a regular expression locating the list in the text — its first "
                "group, or its whole match.", None),
              P("items", "list[string] | object",
                "For `list` and `capture`: the vocabulary, a list or a map with `weights` or `uniform`. "
                "A capture with one both constrains and canonicalises — a text that says `Ana` "
                "captures the `ana` the vocabulary spells, which is what the value is compared "
                "against downstream — and a match outside it is no match. "
                "A stored word list may be given by reference.", None,
                fields=(
                    P("uniform", "list[string]",
                      "The outcomes, weighted equally. Wins over `weights` when "
                      "both are given.",
                      None),
                    P("weights", "map[string, float]",
                      "Outcome → weight, each finite and at least 0: raw corpus "
                      "frequencies, say. A stored word list may be given by "
                      "reference.",
                      None, stored="text/word-list"),
                ), stored="text/word-list"),
              P("count", "int", "For `list`: how many items a valid list has.", None),
          )),
        P("mode", "string",
          "`\"annotate\"`: emit each record with its measures. "
          "`\"corpus\"`: emit one summary record. `\"items\"`: emit one "
          "record per distinct item a `list` measure parsed, and per "
          "distinct word a `lexical` measure counted.",
          "annotate", choices=("annotate", "corpus", "items")),
        P("keep", "bool",
          "In `annotate` mode, carry the whole item (text, trace, metadata) "
          "on each output record rather than only `id`, `coords` and the "
          "measures — so a capture downstream can replay the story it was "
          "labelled on.",
          False),
    ),
    example={
        "measures": [
            {"type": "pattern", "name": "lighthouse",
             "patterns": ["\\blighthouse\\b"], "ignore_case": True},
            {"type": "lexical", "name": "lex"},
        ],
        "keep": True,
    },
)


def run(ctx, inputs, params):
    return build_collection(measure_texts(inputs, params))


_WORD_RE = None


def _split_words(text: str, lowercase: bool, min_length: int) -> list[str]:
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


def _read_vocabulary(name: str, items: Any) -> list[str]:
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


def _read_exclude(name: str, exclude: Any, lowercase: bool) -> frozenset[str]:
    if exclude is None:
        return frozenset()
    if isinstance(exclude, Mapping):
        exclude = exclude.get("words") or list((exclude.get("weights") or {}).keys())
    if not isinstance(exclude, (list, tuple)):
        raise TypeError(
            f"text/measure measure {name!r}: `exclude` is a list of words")
    return frozenset(str(w).lower() if lowercase else str(w) for w in exclude)


def measure_texts(inputs: Mapping[str, Any],
                  params: Mapping[str, Any]) -> Any:
    """text/measure — configurable per-text measurements over a corpus
    of records or a collection of documents (what `text/generate`
    produces): meta-leak counts, opening-phrase counts, lexical spread,
    corpus-frequency of vocabulary, each as a node with lineage.

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
            → per-record parse of a drawn list: items,
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
                  or not `items` contains it.
    """
    import math as _math
    import re

    raw = inputs.get("records")
    if raw is None:
        raw = inputs.get("documents")
    if raw is None:
        raise ValueError(
            "text/measure needs texts on its `records` or `documents` port")
    recs = read_items(raw)
    field = "text"
    measures = params.get("measures") or []
    mode = params.get("mode", "annotate")
    if mode not in ("annotate", "corpus", "items"):
        raise ValueError(
            "text/measure mode must be 'annotate', 'corpus' or 'items', not "
            f"{mode!r}")
    # `keep`: an annotated row carries the whole item —
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
            lower = bool(m.get("lowercase", True))
            compiled.append((name, kind,
                             {"lowercase": lower,
                              "min_length": int(m.get("min_length", 1)),
                              "exclude": _read_exclude(name, m.get("exclude"), lower)}))
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
            # One value, under the name asked for: a `list` measure can
            # read the same thing, but only as the first item of a list,
            # with five columns of list statistics beside it.
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
                           for k in _read_vocabulary(name, vocab)}
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
                "vocab": ({(k.casefold() if fold else k) for k in _read_vocabulary(name, vocab)}
                          if vocab is not None else None),
                "count": int(m["count"]) if m.get("count") is not None else None,
                "fold": fold}))
        else:
            raise ValueError(f"text/measure: unknown measure kind {kind!r}")

    out = []
    corpus_items: dict[str, set[str]] = {}
    # What the corpus said, item by item: the vocabulary labels
    # an answer, it does not decide whether the answer counts.
    said: dict[str, dict[str, dict]] = {}
    captured: dict[str, list[Any]] = {}
    corpus_words: list[str] = []
    # Per lexical measure, word -> [occurrences, texts using it].
    used: dict[str, dict[str, list[int]]] = {}
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
                words = _split_words(text, cfg["lowercase"], cfg["min_length"])
                if cfg["exclude"]:
                    words = [w for w in words if w not in cfg["exclude"]]
                distinct = len(set(words))
                row[f"{name}_words"] = len(words)
                row[f"{name}_distinct"] = distinct
                row[f"{name}_dup"] = (round(1.0 - distinct / len(words), 4)
                                      if words else 0.0)
                corpus_words.extend(words)
                vocabulary = used.setdefault(name, {})
                for w in words:
                    vocabulary.setdefault(w, [0, 0])[0] += 1
                for w in set(words):
                    vocabulary[w][1] += 1
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
                words = _split_words(text, cfg["lowercase"], cfg["min_length"])
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
        if not any(kind in ("list", "lexical") for _, kind, _ in compiled):
            raise ValueError(
                "text/measure mode 'items' needs a `list` or `lexical` measure: "
                "it tallies what the lists said, or the words the texts used")
        rows = []
        for name, tally in said.items():
            total = sum(t["count"] for t in tally.values()) or 1
            for t in sorted(tally.values(), key=lambda t: (-t["count"], t["item"])):
                rows.append({"id": f"{name}:{t['item']}",
                             "coords": {"measure": name, "item": t["item"]},
                             **t, "share": round(t["count"] / total, 6)})
        for name, vocabulary in used.items():
            total = sum(c for c, _ in vocabulary.values()) or 1
            for word, (c, texts) in sorted(vocabulary.items(),
                                           key=lambda kv: (-kv[1][1], -kv[1][0], kv[0])):
                rows.append({"id": f"{name}:{word}",
                             "coords": {"measure": name, "item": word},
                             "item": word, "count": c, "texts": texts,
                             "share": round(c / total, 6)})
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
