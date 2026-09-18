"""Ops over **records**: making them, filtering, joining, summarising and
presenting them. None of these touch a model; they are the plumbing
between the ops that do.

A record is a JSON object with an `id`, usually a `coords` object (the
experimental condition it belongs to — `{"genre": "noir", "seed": 3}`),
and whatever fields the op that made it wrote. Most of the ops here read
records from an edge onto their `records` port.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import WILDCARD, Output, In, Op, P
from mechbench_compute.lexicon.common import TARGET_UNIFORM, TARGET_WEIGHTS

_RECORDS = In("records", "collection | records/table",
              "The records to work on: any collection of items — records, "
              "decision reads, vectors, verdicts, tree summaries — since every "
              "item has an id and its fields; a table's rows are read as records.",
              many=True)

_COORDS_TYPE = "map[string, string | float]"

#: One factor of a crossed design: enumerated levels, sampled ones, or both.
_FACTOR_FIELDS = (
    P("name", "string", "The factor's name: the coordinate every record carries it under."),
    P("levels", "list[object]", "The enumerated levels.", None,
      fields=(
          P("key", "string", "The level's key: its coordinate value, and part of each record's id."),
          P("value", "string", "The level's text, for `records/fill`; the key by default.", None),
          P("coords", _COORDS_TYPE, "Further coordinates the level stamps on its records.", None),
      )),
    P("sampled", "object | list[object]",
      "Levels drawn by a generator, or several generators; each value depends only on the "
      "generator's `seed` and its index.",
      None,
      fields=(
          P("type", "string",
            "`noise` draws random strings; `words` draws word sequences. One of `type` or "
            "`kind` is required.",
            None, choices=("noise", "words")),
          P("kind", "string", "The older spelling of `type`; read when `type` is absent.", None,
            choices=("noise", "words")),
          P("size", "int", "How long each value is: characters for `noise`, words for `words`."),
          P("count", "int", "How many values to draw."),
          P("start", "int", "The first index, so a later node can extend the set.", 0),
          P("seed", "int | string", "The generator's own seed (not the node's).", 0),
          P("word_list", "list[string] | object",
            "For `words`: the words drawn from — inline, or a stored word list by reference.", None,
            fields=(P("words", "list[string]", "The words."),), stored="text/word-list"),
          P("wrap", "string", "A template each value is framed in, `{x}` for the value.", "{x}"),
          P("key_prefix", "string", "The level keys' prefix; `<type>-<size>` by default.", None),
          P("kind_coord", "string",
            "The coordinate naming the generator; `<factor name>_kind` by default.", None),
          P("coords", _COORDS_TYPE, "Further coordinates stamped on the sampled records.", None),
      )),
)

_FACTORS_DESC = """\
Each factor has a `name` and its **levels**: either enumerated —
`{"levels": [{"key": "noir", "value": "a noir story"}, …]}` — or **sampled**
by a generator — `{"sampled": {"type": "noise", "size": 12, "count": 5}}`
(random strings) or `{"type": "words", "size": 3, "count": 5, "word_list":
[…]}` (random word sequences) — or both. A generator's values depend only on
(`seed`, index), so `start`/`count` can grow the set later without changing
what exists; a `wrap` template such as `"Seed: {x}"` frames each value, and a
`key_prefix` names the level keys.

The output is one record per combination: `id` joins the level keys
(`noir-seed-2`), `coords` maps each factor name to its level key (plus any
`coords` the level or generator attached — generators stamp
`<name>_kind`), and `values` maps each factor name to its level's text,
ready for `records/fill`.
"""

FACTOR_CROSS = Op(
    name="records/cross",
    summary=(
        "Make one record per combination of experimental factors — the "
        "fully-crossed design, with each record carrying its coordinates."
    ),
    description=_FACTORS_DESC,
    inputs=(),
    output=Output('records/record', collection=True, doc='One record per combination: `{id, coords, values}`.'),
    params=(
        P("factors", "list[object]",
          "The factors to cross, each `{name, levels?, sampled?}` as "
          "described above. At least one is needed for a non-trivial "
          "design.",
          None, fields=_FACTOR_FIELDS),
        P("axes", "list[object]",
          "The older name for `factors`; read when `factors` is absent.",
          None, fields=_FACTOR_FIELDS),
    ),
    example={
        "factors": [
            {"name": "genre", "levels": [
                {"key": "noir", "value": "a noir detective story"},
                {"key": "fable", "value": "a fable with a moral"},
            ]},
            {"name": "seed", "sampled": {"type": "noise", "size": 8, "count": 4}},
        ],
    },
)

TEMPLATE = Op(
    name="records/fill",
    summary=(
        "Fill named string templates from each record's factor values — "
        "turn a design into prompts."
    ),
    description="""\
For every record, each entry of `templates` becomes a field of the same
name with `{factor}` placeholders replaced by the record's `values`. A
substituted value may itself contain placeholders (an elaborate opening that
embeds `{gender}`); substitution repeats until nothing changes, up to four
passes. Everything outside braces is verbatim.

The output records keep their `id` and `coords`. Name the templates after
the fields the next op reads — `system`, `user`, `prefill` for the
chat-shaped ops — and no adaptation step is needed between them.
""",
    inputs=(
        In("records", "records/record",
           "Records with `values`, usually from `records/cross`.", many=True),
    ),
    output=Output('records/record', collection=True, doc='One record per input record: `{id, coords}` plus one field per template.'),
    params=(
        P("templates", "map[string, string]",
          "Field name → template string. `{name}` is replaced by the "
          "record's value for factor `name`.",
          None),
    ),
    example={
        "templates": {
            "system": "You are a storyteller.",
            "user": "Write {genre}. Begin with the phrase: {seed}",
        },
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/design"}}},
)

RENAME = Op(
    name="records/rename",
    summary=(
        "Rename fields on every record — the one visible adaptation step "
        "between an op that wrote a field under one name and an op that "
        "reads it under another."
    ),
    description="""\
Every op reads the fields it names: the chat-shaped ops read `system`,
`user` and `prefill`; `eval/score` reads `prediction` and `reference`;
`text/measure` and `eval/judge` read `text`. When a record carries the right
value under another name, this op moves it, and the graph shows the move
rather than hiding it in a parameter.

`fields` maps old name → new name. A name may be a dotted path, so a
value can be moved into or out of `coords` (`{"opening": "coords.opening"}`
makes a measurement a coordinate the grouping ops can read) or lifted from
a document's `metadata.coords`. A record without the old field is left as
it is. Everything not named is kept.
""",
    inputs=(_RECORDS,),
    output=Output('records/record', collection=True, doc='The same records, with the named fields moved.'),
    params=(
        P("fields", "map[string, string]",
          "Old name → new name, each a field or a dotted path such as "
          "`coords.genre`."),
    ),
    example={"fields": {"question": "user", "opening": "coords.opening"}},
    example_inputs={"records": {"$ref": {"bench": "you/lab/records"}}},
)

SELECT = Op(
    name="records/select",
    summary=(
        "Keep the records that match a set of field values, and optionally "
        "keep only some of their fields."
    ),
    description="""\
`where` is a map of field → value (or list of acceptable values). A key is
read from the record's `coords` when it is a coordinate, and from the
record itself otherwise — so a field written by `text/measure` (a pattern hit)
filters as easily as a design coordinate. A record passes when every key
matches.

`fields` projects the survivors down to `id`, `coords` and the named fields.
""",
    inputs=(_RECORDS,),
    output=Output('records/record', collection=True, doc='The matching records.'),
    params=(
        P("where", "map[string, string | float | bool | list[string | float | bool]]",
          "Field → value or list of values. `{\"genre\": \"noir\", "
          "\"leak\": 0}` keeps noir records with no leak.",
          None),
        P("fields", "list[string]",
          "Keep only these fields (plus `id` and `coords`). By default the "
          "whole record is kept.",
          None),
    ),
    example={"where": {"genre": ["noir", "fable"], "leak": 0}},
    example_inputs={"records": {"$ref": {"bench": "you/lab/records"}}},
)

UNION = Op(
    name="records/union",
    summary=(
        "Concatenate several record streams into one, stamping each record "
        "with the port it came from — collections grow by union, never by "
        "mutation."
    ),
    description="""\
Every edge into the node is one input; its port name becomes the record's
value on the `batch_axis` coordinate, so the source of every record stays
visible downstream. `segments` records how many came from each port.

A union of vector collections stays a vector collection: items from a base
capture and an adapted capture become one collection whose items carry the
port they came from on the `batch_axis` coordinate — the grouping
`direction/fit` reads with `axis` set to it. Cross-model comparison
is a union followed by the direction algebra.
""",
    inputs=(
        In(WILDCARD, "collection",
           "Any number of edges, each carrying a collection — of records, or "
           "of `activations/vector` — on a port of your naming; the name "
           "becomes the value on the batch coordinate.", many=True),
    ),
    output=(
        Output('records/record', collection=True, doc="Every input's records, each with the batch coordinate; the header's `segments` says how many came from each port. When every input was a collection of `activations/vector`, so is the output, every item keeping its own `space`.")
    ),
    params=(
        P("batch_axis", "string",
          "The coordinate each record gains, set to the name of the port it "
          "arrived on.",
          "batch"),
    ),
    example={"batch_axis": "run"},
    example_inputs={"base": {"$ref": {"bench": "you/lab/base_vectors"}}, "adapted": {"$ref": {"bench": "you/lab/adapted_vectors"}}},
)

ZIP = Op(
    name="records/zip",
    summary=(
        "Align several branches' records into one record per key — record "
        "7 of each branch together — keeping which branch each came from."
    ),
    description="""\
Two branches over the same prompts produce two streams, and a node that
compares them needs their records paired, not concatenated. `union` puts
both streams in one collection and marks where each came from; this puts
each branch's record for a key IN one record.

The **key** is `id` by default, or a list of coordinate names — `by:
["prompt", "seed"]` — which is what to use when two branches number their
records differently but share a design. Two records with the same key in
one branch are refused: zip needs one per key, and the fix is usually to
key on more coordinates.

Branches arrive on the **variadic `branches` port**: one edge per branch,
each named by the node it came from unless `names` says otherwise. The
port is ordered, so `names` lines up with the edges as the graph declares
them.

A key that is not in every branch is an error by default, because a
silently shorter output is a silently different experiment. `drop` keeps
the keys every branch has; `placeholder` keeps them all and marks what is
absent, for a readout that can report a missing arm.
""",
    inputs=(
        In("branches", "collection",
           "One edge per branch, each a collection of records. Ordered: the "
           "first edge is the first branch.",
           many=True, variadic=True, min_edges=2),
    ),
    output=Output('records/record', collection=True, doc="One record per key: `id`, `coords` from the first branch that has it, and `branches` — a map of branch name to that branch's record (or `{missing: true}` under `placeholder`). With `flatten`, each branch's fields are copied up under a `<branch>_` prefix instead. The header carries `branches` (name, source node, count) and `zipped` (the key, how many came out, the policy, and what was dropped)."),
    params=(
        P("by", "\"id\" | list[string]",
          "What to align on: record ids, or the named coordinates.",
          "id"),
        P("on_mismatch", "string",
          "A key missing from some branch: `\"fail\"`, `\"drop\"` (keep only "
          "the keys every branch has) or `\"placeholder\"` (keep them all, "
          "marking what is absent).",
          "fail", choices=("fail", "drop", "placeholder")),
        P("names", "list[string]",
          "What to call each branch, in edge order. Defaults to the source "
          "node ids.",
          None),
        P("flatten", "bool",
          "Copy each branch's fields up under a `<branch>_` prefix instead "
          "of nesting them under `branches`.",
          False),
    ),
    example={"by": ["prompt"], "names": ["base", "adapted"]},
    example_inputs={"branches": {"$ref": {"bench": "you/lab/base_reads"}}},
)

MAP = Op(
    name="records/map",
    summary=(
        "Run a whole sub-protocol once per record — the fan-out between a "
        "node that loops over its own items and a run set that loops over "
        "whole runs."
    ),
    description="""\
A run set fans out over runs; a node fans out over the items inside it.
Between the two there was nothing — so a graph that wanted to do several
things to each record had to be written once per record, or flattened
into one node that knew how to do all of them.

`body` is a graph, run once per record with that record's fields bound
into its holes by `bind`: `{"topic": "user"}` puts each record's `user`
field in the body's `$topic`. The body is written once, against one
record, and reads as what it is.

Each invocation is an **item keyed by the record's id**, so an
interrupted map resumes exactly as an interrupted chat node does: the
records already done are reused, the rest are run. The body sees one
record and nothing else, which is also why chunking a map is the same
map: there is no state between records to lose.

`collect` says what comes back. `stream` (the default) flattens every
invocation's output into one collection, each item's id prefixed with
its record's and carrying a `mapped` coordinate; `first` keeps one item
per record; `all` keeps each invocation's items nested under its record.
When the body ends at more than one node, `output` names the one to
collect.
""",
    inputs=(
        In("records", "records/record",
           "The stream to map over: one invocation of the body per record.",
           many=True),
    ),
    output=Output('records/record', collection=True, doc="Under `stream`, every invocation's items in one collection, each id prefixed `<record>:<item>` and carrying `coords.mapped`; under `first` or `all`, one item per record. The header's `mapped` says how many records ran, under which policy, and what the body's nodes were."),
    params=(
        P("body", "object",
          "The graph to run per record — `{nodes, edges}`, the same shape a "
          "protocol's graph has. Its holes are filled by `bind`.", fields=(
              P("nodes", "list[json]", "The body's nodes, as a protocol graph writes them."),
              P("edges", "list[json]", "The body's edges.", []),
          )),
        P("bind", "map[string, string]",
          "Hole name → the record field that fills it, per record.",
          None),
        P("collect", "string",
          "`\"stream\"` (flatten every invocation's items), `\"first\"` "
          "(one item per record) or `\"all\"` (nest each invocation's items "
          "under its record).",
          "stream", choices=("stream", "first", "all")),
        P("output", "string",
          "Which of the body's terminal nodes to collect, when it has more "
          "than one.",
          None),
    ),
    example={"bind": {"topic": "user"}, "collect": "stream",
             "body": {"nodes": [{"id": "write", "block": "text/generate",
                                 "params": {"model": {"$param": "model"}, "n": 3,
                                            "messages": [{"role": "user",
                                                          "content": "$topic"}]}}],
                      "edges": []}},
    example_inputs={"records": {"$ref": {"bench": "you/lab/topics"}}},
)

PAIRED_DELTA = Op(
    name="records/subtract",
    summary=(
        "Subtract a matched baseline from every record — the treatment "
        "effect per condition, ready to summarise."
    ),
    description="""\
Records whose `coords` match `baseline_where` are the baselines. Every other
record finds the baseline that agrees with it on the `match_on` coordinates
and reports its `value` field, the baseline's, and the difference. A record
with no matching baseline is an error, not a silent omission.

The output keeps `coords`, so it feeds `records/summarize` directly.
""",
    inputs=(_RECORDS,),
    output=Output('records/record', collection=True, doc='One record per non-baseline record: `{id, coords, value, baseline, delta}`.'),
    params=(
        P("value", "string",
          "The numeric field to difference. A record has many numeric "
          "fields; this names the one the question is about."),
        P("baseline_where", "map[string, string | float]",
          "Coordinates identifying the baseline records, e.g. "
          "`{\"alpha\": 0}`."),
        P("match_on", "list[string]",
          "Coordinates a record and its baseline must agree on. Empty "
          "means one baseline for everything.",
          None),
    ),
    example={"value": "entropy_bits", "baseline_where": {"alpha": 0}, "match_on": ["prompt"]},
    example_inputs={"records": {"$ref": {"bench": "you/lab/reads"}}},
)

GROUP_STATS = Op(
    name="records/summarize",
    summary=(
        "Group records by coordinates and summarise a numeric field — count, "
        "median, mean, min, max and the share below zero — as a table."
    ),
    description="""\
One row per distinct combination of the `by` coordinates (or one row in all
when `by` is empty). `share_negative` is the fraction of values below zero,
useful when the field is a delta.

A record without the value field is refused by name, because a mean over
"the records that happened to have it" is the kind of number nobody
notices is wrong. When absent values are expected — a judge that could not
be read, an unscored item — set `on_missing: "skip"` and the count of
skipped records is reported on the table as `n_missing`.
""",
    inputs=(_RECORDS,),
    output=(
        Output('records/table', collection=False, doc='One row per group with the `by` coordinates and `n`, `median`, `mean`, `min`, `max`, `share_negative`; `n_missing` when any were skipped.')
    ),
    params=(
        P("value", "string",
          "The numeric field to summarise. A record has many numeric "
          "fields; this names the one the question is about."),
        P("by", "list[string]",
          "The coordinates to group on. Empty gives one overall row.",
          None),
        P("on_missing", "string",
          "`\"error\"`: refuse a record without the field. `\"skip\"`: omit "
          "it and report how many were omitted.",
          "error", choices=("error", "skip")),
    ),
    example={"value": "delta", "by": ["genre", "alpha"]},
    example_inputs={"records": {"$ref": {"bench": "you/lab/deltas"}}},
)

TABLE_FROM_RECORDS = Op(
    name="records/tabulate",
    summary=(
        "Present a record list as a table — coordinates become the leading "
        "columns, scalar fields follow."
    ),
    description="""\
The generic records-to-table step. Every coordinate seen across the records
becomes a column, then every scalar (number or string) field; each column's
type is inferred from its values. Nested fields are left out.
""",
    inputs=(_RECORDS,),
    output=Output('records/table', collection=False, doc='`columns` (`{name, dtype}`) and `rows`.'),
    params=(
        P("row_axis", "string",
          "What one row stands for, recorded on the table for its renderer "
          "— `\"record\"`, `\"condition\"`, `\"layer\"`.",
          "record"),
    ),
    example={"row_axis": "condition", "name": "steering deltas"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/deltas"}}},
)

TEXT_STATS = Op(
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
| `lexical` | `<name>_words`, `<name>_distinct`, `<name>_dup` (1 − distinct/words) | `lowercase` (default true), `min_length` |
| `corpus_frequency` | `<name>`: the statistic over the reference frequency of the text's words; `<name>_coverage`: the fraction of words found in the table | `frequencies` (word → count, or wire a `frequencies` input), `stat`: `"mean_log10"` (rarer vocabulary ⇒ lower), `"mean"` or `"coverage"`, `lowercase`, `min_length` |
| `list` | `<name>_parsed` (1 if the list was found), `<name>_items`, `<name>_distinct`, `<name>_duplicates`, `<name>_unknown` (items outside `items`, when given), `<name>_first`, `<name>_valid` (found, no duplicates, nothing unknown, and `count` items when given) | `separator` (default `", "`), `extract` (a regex whose first group is the list; the whole text without it), `items` (the vocabulary: a list, or a map's `weights` or `uniform`), `count`, `ignore_case` |

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
                choices=("pattern", "lexical", "corpus_frequency", "list")),
              P("kind", "string", "The older spelling of `type`; read when `type` is absent.", None,
                choices=("pattern", "lexical", "corpus_frequency", "list")),
              P("name", "string", "The field it writes; the type by default.", None),
              P("patterns", "list[string]", "For `pattern`: the regular expressions.", None),
              P("where", "string", "For `pattern`: match anywhere, or only at the start.", "anywhere",
                choices=("anywhere", "prefix")),
              P("ignore_case", "bool", "For `pattern` and `list`: match without regard to case.", False),
              P("lowercase", "bool", "For `lexical` and `corpus_frequency`: lowercase words first.", True),
              P("min_length", "int", "For `lexical` and `corpus_frequency`: the shortest word counted.", 1),
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
                "For `list`: the vocabulary, a list or a map with `weights` or `uniform`. "
                "A stored word list may be given by reference.", None,
                fields=(TARGET_UNIFORM, TARGET_WEIGHTS), stored="text/word-list"),
              P("count", "int", "For `list`: how many items a valid list has.", None),
          )),
        P("mode", "string",
          "`\"annotate\"`: emit each record with its measures. "
          "`\"corpus\"`: emit one summary record. `\"items\"`: emit one "
          "record per distinct item a `list` measure parsed.",
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

REDUCE_SUM = Op(
    name="records/total",
    summary="The exact sum of a numeric field over all records, with the count.",
    description="""\
An exact reduce: values are kept as a multiset and summed with a correctly
rounded algorithm, so the result is the same whatever order or chunking the
records arrived in. Safe to run over partial results and merge.
""",
    inputs=(_RECORDS,),
    output=Output('records/sum', collection=False, doc='`{n, sum}`.'),
    params=(P("value", "string", "The numeric field to sum."),),
    example={"value": "cost_usd"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/records"}}},
)

REDUCE_TOP_K = Op(
    name="records/rank",
    summary="The k records with the largest value of a field.",
    description="""\
Sorted by the field descending, ties broken by `id`, so the result is
deterministic. An exact reduce: the top-k of a union is the top-k of the
top-ks, so partial results merge without loss.
""",
    inputs=(_RECORDS,),
    output=Output('records/record', collection=True, doc='The top k, in order.'),
    params=(
        P("value", "string", "The numeric field to rank by."),
        P("k", "int", "How many to keep.", 10),
    ),
    example={"value": "delta", "k": 5},
    example_inputs={"records": {"$ref": {"bench": "you/lab/deltas"}}},
)

REDUCE_HISTOGRAM = Op(
    name="records/bin",
    summary="Count a numeric field into fixed, equal-width bins.",
    description="""\
`bins` equal-width bins span `[lo, hi)`; values below `lo` and at or above
`hi` are counted separately rather than dropped, so the total always equals
the number of records. An exact reduce: counts add.
""",
    inputs=(_RECORDS,),
    output=Output('records/histogram', collection=False, doc='`bins` (the counts, in order), `below`, `above`.'),
    params=(
        P("value", "string", "The numeric field to bin."),
        P("lo", "float", "The lower edge of the first bin."),
        P("hi", "float", "The upper edge of the last bin (exclusive)."),
        P("bins", "int", "How many equal-width bins between `lo` and `hi`."),
    ),
    example={"value": "entropy_bits", "lo": 0.0, "hi": 8.0, "bins": 16},
    example_inputs={"records": {"$ref": {"bench": "you/lab/reads"}}},
)

EVAL_EXPECTATION = Op(
    name="eval/expect",
    summary=(
        "Judge each decision read against an expectation carried as data — "
        "uniform over the outcomes, a required answer, a minimum entropy, or "
        "a target distribution — and report pass/fail with the rate."
    ),
    description="""\
Results (from `logits/read`) and expectations are joined on `id`. Each
expectation record has an `expect` object:

| `type` | Passes when | Fields |
|---|---|---|
| `uniform` | the KL divergence from uniform over the named outcomes is at most `max_kl_bits` | `over` (the outcomes), `max_kl_bits` (default 0.1) |
| `weights` | the KL divergence from the normalised `weights` is at most `max_kl_bits` | `weights` (outcome → weight), `max_kl_bits` |
| `answer` | the expected token's probability is at least `min_p` | `value`, `min_p` (default 0.99) |
| `min_entropy` | the read's entropy is at least `bits` | `bits` |
| `absent` | the total mass on the named outcomes is at most `max_p` — outcomes that should not be said, such as the genres already in a list | `over` (the outcomes), `max_p` (default 0.01) |

Outcome masses come from the read's `tracked` (each outcome by its own
name, a token or a `complete` outcome), else from its `top` by exact token
text. A read with no mass on any outcome is reported as *unjudgeable*
(`pass: null` with a `note`) rather than counted as a failure — a hole in
the read must not masquerade as a verdict. For `absent` the bar is higher:
every named outcome must be in `tracked`, since an outcome the read never
scored has no mass to report, not a mass of zero.

The header's `summary` carries the pass rate: the number a write-up cites.
""",
    inputs=(
        In("results", "logits/distribution",
           "The decision reads — a collection of `logits/decision`, or of any "
           "kind that extends `logits/distribution`.", many=True),
        In("expectations", "records/record",
           "Records `{id, expect}`, one per result to judge.", many=True),
    ),
    output=(
        Output('eval/verdict', collection=True, doc='One verdict per judged result: `id`, `coords`, `expect`, `entropy_bits`, `kl_bits`, `mass`, `p_expected`, and `pass` (null with a `note` when unjudgeable). The header\'s `summary` carries `pass_rate`, `n_pass`, `n_judged` and `n_unjudgeable`.')
    ),
    params=(),
    example={},
    example_inputs={
        "results": {"$ref": {"bench": "you/lab/reads"}},
        "expectations": [
            {"id": "d6", "expect": {"type": "uniform",
                                    "over": ["1", "2", "3", "4", "5", "6"],
                                    "max_kl_bits": 0.1}},
        ],
    },
)

VIZ_SPEC = Op(
    name="records/plot",
    summary=(
        "Describe a chart of an upstream table as a stored object — what to "
        "plot on which axes — so it renders beside the data and re-renders "
        "when the data changes."
    ),
    description="""\
The spec names a mark and an encoding. When the executor knows the input's
stored location, the spec references it (`source`) and the chart is drawn
from the live data; otherwise the rows ride inline under `data.rows` and the
spec is self-contained. Coordinates are flattened into each row so they can
be encoded directly.
""",
    inputs=(
        In("records", "collection | records/table",
           "The table, or any collection of items, to chart.", many=True),
    ),
    output=Output('records/chart', collection=False, doc='`title`, `mark`, `encoding` (`x`, `y`, `series`), and `source` or `data`.'),
    params=(
        P("encoding", "object",
          "`{\"x\": field, \"y\": field, \"series\": field}` — which fields "
          "go on which axis, and which splits the data into series.",
          None, fields=(
              P("x", "string", "The field on the x axis.", None),
              P("y", "string", "The field on the y axis.", None),
              P("series", "string", "The field that splits the rows into series.", None),
          )),
        P("x", "string", "The x field; the same as `encoding.x`.", None),
        P("y", "string", "The y field; the same as `encoding.y`.", None),
        P("mark", "string", "The mark: `\"bar\"`, `\"line\"`, `\"point\"`.", "bar", choices=("bar", "line", "point")),
        P("title", "string", "The chart's title.", ""),
    ),
    example={
        "title": "Δ log p by layer",
        "mark": "line",
        "encoding": {"x": "layer", "y": "mean", "series": "genre"},
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/table"}}},
)

VECTORS_SIMILARITY = Op(
    name="geometry/compare",
    summary=(
        "The pairwise matrix of a collection's items under a metric their "
        "kind declares — cosine over vectors, Jensen–Shannon over decision "
        "reads, hamming over records — with how well the groups separate."
    ),
    description="""\
A kind declares how its items compare, the way it declares how they are
drawn: `activations/vector` (and so directions and trajectory points) by
`cosine` (option `center`), `euclidean` or `dot`; `logits/distribution`
(and so decision reads, funnels and readouts) by `jensen-shannon`,
`hellinger`, `total-variation` or `kl`; `records/record` (and every record
kind) by `hamming` over `coords`. The op takes any such collection on
`items`, applies the named `metric` — the kind's first when none is named —
and produces the matrix with the metric, its `options`, and whether it is
symmetric recorded on the header.

Items are grouped `by` a header axis before comparing: `"space"` (the
default for vectors) compares only items from one layer and head; a
coordinate name (`"layer"` for a funnel, `"factor"` for a readout) compares
within each of its values; `null` compares everything at once. One item per
group, with every pair listed for groups of at most thirty-two.

When every item in a group has a value on the `axis` coordinate and there is
more than one value, the item also reports the mean value within groups,
between groups, their gap, the fraction of items whose nearest neighbour
shares their value, and the silhouette score.

Two vectors compare only within one space; two decision reads compare over
the union of the tokens they carry, the mass neither names counted as one
last bucket. Raw cosine between transformer activations is dominated by a
shared direction they all lean toward; for a variety measure use `cosine`
with `options: {"center": true}`, which subtracts it, and `geometry/span`
downstream.
""",
    inputs=(
        In("items", "activations/vector | logits/distribution | records/record",
           "The items to compare: any collection whose kind declares metrics.",
           many=True),
    ),
    output=(
        Output('geometry/similarity', collection=True, doc='One item per group: `{group, layer?, head?, space?, ids, labels, matrix, pairs?, separation?, nn_purity?, silhouette?}`, `labels` being the items\' values on the `axis` coordinate. The header carries `metric`, `metric_kind`, `symmetric`, `options`, `over` (the kind compared), `by`, `axis`, and `position`/`point` for vectors.')
    ),
    params=(
        P("metric", "string",
          "Which of the kind's metrics to apply. By default the kind's first: "
          "`cosine` for vectors, `jensen-shannon` for distributions, "
          "`hamming` for records.",
          None),
        P("options", "map[string, json]",
          "The metric's options, as it declares them — `{\"center\": true}` "
          "for `cosine`.",
          None),
        P("by", "string | null",
          "The header axis to group on before comparing: `\"space\"` (per "
          "layer and head), a coordinate name, or `null` for one group. "
          "Defaults to `\"space\"` when the items carry one.",
          None),
        P("axis", "string",
          "The coordinate the separation reads. `label` reads the older "
          "`label` field as well.",
          "label"),
    ),
    example={"metric": "cosine", "options": {"center": True}, "axis": "genre"},
    example_inputs={"items": {"$ref": {"bench": "you/lab/vectors"}}},
)

VECTORS_MST = Op(
    name="geometry/span",
    summary=(
        "Measure how varied a set of items is with a minimum spanning tree "
        "over their pairwise distances — the spread, its clumpiness, and how "
        "many clusters the bridges imply — for anything a metric compares."
    ),
    description="""\
Average pairwise distance cannot tell three tight clumps from an even
spread; a minimum spanning tree can. Its edges are the cheapest set that
still connects every point, so within a clump edges are short and between
clumps there is one long **bridge**. Per group the block reports the mean
edge (the scale of the spread — small means collapsed), the variance and
coefficient of variation (clumpiness), and the number of bridges — edges
more than `bridge_sigma` standard deviations above the mean — which is
roughly the cluster count minus one. Read `mean` and `variance` together:
a collapsed corpus and an evenly varied one both have low variance, for
opposite reasons.

The tree is built on a `geometry/similarity` collection, so it stands over
whatever that op compared: a corpus of vectors by centred cosine, the
adapters' axes, a grid of decision reads by Jensen–Shannon, the design's
records by how many factors differ. A distance metric is used as it is; a
cosine as `1 − cosine`; any other similarity as `max − value`. A metric
that is not symmetric (`kl`) is refused by name. The tree is built
deterministically (ties break toward the lower index), so a run that
reproduces its numbers reproduces its tree.
""",
    inputs=(
        In("similarity", "geometry/similarity",
           "The pairwise matrices, one per group, from `geometry/compare`.",
           many=True),
    ),
    output=Output('geometry/mst', collection=True, doc='One item per group: `group`, `layer`/`head` when the group is a space, `n`, `n_edges`, `mean`, `variance`, `stdev`, `cv`, `total`, `min`, `max`, `bridge_threshold`, `bridges`, `components_after_cut`, `ids`, `labels`, and `edges` as `[i, j, weight]` when kept. The header carries `metric`, `options`, `over` (the kind compared), `bridge_sigma` and `axis`. `records/table` reads the items as its rows.'),
    params=(
        P("bridge_sigma", "float",
          "How many standard deviations above the mean edge an edge must be "
          "to count as a bridge between clusters.",
          2.0),
        P("keep_edges", "bool",
          "Include every tree edge (`[i, j, weight]`) in the output, not "
          "only the statistics.",
          True),
    ),
    example={"bridge_sigma": 2.0},
    example_inputs={"similarity": {"$ref": {"bench": "you/lab/similarity"}}},
)

OPS: tuple[Op, ...] = (
    FACTOR_CROSS, TEMPLATE, RENAME, SELECT, UNION, ZIP, MAP, PAIRED_DELTA,
    GROUP_STATS,
    TABLE_FROM_RECORDS, TEXT_STATS, REDUCE_SUM, REDUCE_TOP_K, REDUCE_HISTOGRAM,
    EVAL_EXPECTATION, VIZ_SPEC, VECTORS_SIMILARITY, VECTORS_MST,
)
