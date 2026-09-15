"""Ops over **records**: making them, filtering, joining, summarising and
presenting them. None of these touch a model; they are the plumbing
between the ops that do.

A record is a JSON object with an `id`, usually a `coords` object (the
experimental condition it belongs to — `{"genre": "noir", "seed": 3}`),
and whatever fields the op that made it wrote. Most of the ops here read
records from an edge onto their `records` port.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import WILDCARD, Emits, In, Op, P

_RECORDS = In("records", "records/record | records/table",
              "The records to work on. A table's rows are read as records.",
              many=True)

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
ready for `records/template`.
"""

FACTOR_CROSS = Op(
    name="records/cross",
    summary=(
        "Make one record per combination of experimental factors — the "
        "fully-crossed design, with each record carrying its coordinates."
    ),
    description=_FACTORS_DESC,
    inputs=(),
    emits=Emits('records/record', collection=True, doc='One record per combination: `{id, coords, values}`.'),
    params=(
        P("factors", "list[object]",
          "The factors to cross, each `{name, levels?, sampled?}` as "
          "described above. At least one is needed for a non-trivial "
          "design.",
          None),
        P("axes", "list[object]",
          "The older name for `factors`; read when `factors` is absent.",
          None),
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
    name="records/template",
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
    emits=Emits('records/record', collection=True, doc='One record per input record: `{id, coords}` plus one field per template.'),
    params=(
        P("templates", "object",
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
    example_inputs={"records": {"$fetch": "$design"}},
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
`user` and `prefill`; `eval/metric` reads `prediction` and `reference`;
`text/stats` and `eval/judge` read `text`. When a record carries the right
value under another name, this op moves it, and the graph shows the move
rather than hiding it in a parameter.

`fields` maps old name → new name. A name may be a dotted path, so a
value can be moved into or out of `coords` (`{"opening": "coords.opening"}`
makes a measurement a coordinate the grouping ops can read) or lifted from
a document's `metadata.coords`. A record without the old field is left as
it is. Everything not named is kept.
""",
    inputs=(_RECORDS,),
    emits=Emits('records/record', collection=True, doc='The same records, with the named fields moved.'),
    params=(
        P("fields", "object",
          "Old name → new name, each a field or a dotted path such as "
          "`coords.genre`."),
    ),
    example={"fields": {"question": "user", "opening": "coords.opening"}},
    example_inputs={"records": {"$fetch": "$records"}},
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
record itself otherwise — so a field written by `text/stats` (a pattern hit)
filters as easily as a design coordinate. A record passes when every key
matches.

`fields` projects the survivors down to `id`, `coords` and the named fields.
""",
    inputs=(_RECORDS,),
    emits=Emits('records/record', collection=True, doc='The matching records.'),
    params=(
        P("where", "object",
          "Field → value or list of values. `{\"genre\": \"noir\", "
          "\"leak\": 0}` keeps noir records with no leak.",
          None),
        P("fields", "list[string]",
          "Keep only these fields (plus `id` and `coords`). By default the "
          "whole record is kept.",
          None),
    ),
    example={"where": {"genre": ["noir", "fable"], "leak": 0}},
    example_inputs={"records": {"$fetch": "$records"}},
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
`direction/from-vectors` reads with `axis` set to it. Cross-model comparison
is a union followed by the direction algebra.
""",
    inputs=(
        In(WILDCARD, "collection",
           "Any number of edges, each carrying a collection — of records, or "
           "of `activations/vector` — on a port of your naming; the name "
           "becomes the value on the batch coordinate.", many=True),
    ),
    emits=(
        Emits('records/record', collection=True, doc="Every input's records, each with the batch coordinate; the header's `segments` says how many came from each port. When every input was a collection of `activations/vector`, so is the output, every item keeping its own `space`.")
    ),
    params=(
        P("batch_axis", "string",
          "The coordinate each record gains, set to the name of the port it "
          "arrived on.",
          "batch"),
    ),
    example={"batch_axis": "run"},
    example_inputs={"base": {"$fetch": "$base_vectors"}, "adapted": {"$fetch": "$adapted_vectors"}},
)

PAIRED_DELTA = Op(
    name="records/delta",
    summary=(
        "Subtract a matched baseline from every record — the treatment "
        "effect per condition, ready to summarise."
    ),
    description="""\
Records whose `coords` match `baseline_where` are the baselines. Every other
record finds the baseline that agrees with it on the `match_on` coordinates
and reports its `value` field, the baseline's, and the difference. A record
with no matching baseline is an error, not a silent omission.

The output keeps `coords`, so it feeds `records/stats` directly.
""",
    inputs=(_RECORDS,),
    emits=Emits('records/record', collection=True, doc='One record per non-baseline record: `{id, coords, value, baseline, delta}`.'),
    params=(
        P("value", "string",
          "The numeric field to difference. A record has many numeric "
          "fields; this names the one the question is about."),
        P("baseline_where", "object",
          "Coordinates identifying the baseline records, e.g. "
          "`{\"alpha\": 0}`."),
        P("match_on", "list[string]",
          "Coordinates a record and its baseline must agree on. Empty "
          "means one baseline for everything.",
          None),
    ),
    example={"value": "entropy_bits", "baseline_where": {"alpha": 0}, "match_on": ["prompt"]},
    example_inputs={"records": {"$fetch": "$reads"}},
)

GROUP_STATS = Op(
    name="records/stats",
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
    emits=(
        Emits('records/table', collection=False, doc='One row per group with the `by` coordinates and `n`, `median`, `mean`, `min`, `max`, `share_negative`; `n_missing` when any were skipped.')
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
          "error"),
    ),
    example={"value": "delta", "by": ["genre", "alpha"]},
    example_inputs={"records": {"$fetch": "$deltas"}},
)

TABLE_FROM_RECORDS = Op(
    name="records/table",
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
    emits=Emits('records/table', collection=False, doc='`columns` (`{name, dtype}`) and `rows`.'),
    params=(
        P("row_axis", "string",
          "What one row stands for, recorded on the table for its renderer "
          "— `\"record\"`, `\"condition\"`, `\"layer\"`.",
          "record"),
    ),
    example={"row_axis": "condition", "name": "steering deltas"},
    example_inputs={"records": {"$fetch": "$deltas"}},
)

TEXT_STATS = Op(
    name="text/stats",
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

In `annotate` mode the output is the records with those fields added —
ready for `records/select`, `records/stats` or `trajectory/capture`. To
group on a measure downstream, `records/rename` it into `coords`. In
`corpus` mode it is one summary record: per pattern a count and rate,
corpus-wide word and distinct-word counts and duplication, and the mean of
each frequency statistic.
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
    emits=(
        Emits('records/record', collection=True, doc='In `annotate` mode, one record per item (`id`, `coords`, the measure fields, and the whole item when `keep` is set). In `corpus` mode, a single record with the corpus summary.')
    ),
    params=(
        P("measures", "list[object]",
          "The measurements to make, each `{kind, name, …}` as in the "
          "table above.",
          None),
        P("mode", "string",
          "`\"annotate\"`: emit each record with its measures. "
          "`\"corpus\"`: emit one summary record.",
          "annotate"),
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
    name="records/sum",
    summary="The exact sum of a numeric field over all records, with the count.",
    description="""\
An exact reduce: values are kept as a multiset and summed with a correctly
rounded algorithm, so the result is the same whatever order or chunking the
records arrived in. Safe to run over partial results and merge.
""",
    inputs=(_RECORDS,),
    emits=Emits('records/sum', collection=False, doc='`{n, sum}`.'),
    params=(P("value", "string", "The numeric field to sum."),),
    example={"value": "cost_usd"},
    example_inputs={"records": {"$fetch": "$records"}},
)

REDUCE_TOP_K = Op(
    name="records/top-k",
    summary="The k records with the largest value of a field.",
    description="""\
Sorted by the field descending, ties broken by `id`, so the result is
deterministic. An exact reduce: the top-k of a union is the top-k of the
top-ks, so partial results merge without loss.
""",
    inputs=(_RECORDS,),
    emits=Emits('records/record', collection=True, doc='The top k, in order.'),
    params=(
        P("value", "string", "The numeric field to rank by."),
        P("k", "int", "How many to keep.", 10),
    ),
    example={"value": "delta", "k": 5},
    example_inputs={"records": {"$fetch": "$deltas"}},
)

REDUCE_HISTOGRAM = Op(
    name="records/histogram",
    summary="Count a numeric field into fixed, equal-width bins.",
    description="""\
`bins` equal-width bins span `[lo, hi)`; values below `lo` and at or above
`hi` are counted separately rather than dropped, so the total always equals
the number of records. An exact reduce: counts add.
""",
    inputs=(_RECORDS,),
    emits=Emits('records/histogram', collection=False, doc='`bins` (the counts, in order), `below`, `above`.'),
    params=(
        P("value", "string", "The numeric field to bin."),
        P("lo", "float", "The lower edge of the first bin."),
        P("hi", "float", "The upper edge of the last bin (exclusive)."),
        P("bins", "int", "How many equal-width bins between `lo` and `hi`."),
    ),
    example={"value": "entropy_bits", "lo": 0.0, "hi": 8.0, "bins": 16},
    example_inputs={"records": {"$fetch": "$reads"}},
)

EVAL_EXPECTATION = Op(
    name="eval/expectation",
    summary=(
        "Judge each decision read against an expectation carried as data — "
        "uniform over the outcomes, a required answer, a minimum entropy, or "
        "a target distribution — and report pass/fail with the rate."
    ),
    description="""\
Results (from `logits/decision`) and expectations are joined on `id`. Each
expectation record has an `expect` object:

| `type` | Passes when | Fields |
|---|---|---|
| `uniform` | the KL divergence from uniform over the named outcomes is at most `max_kl_bits` | `over` (the outcomes), `max_kl_bits` (default 0.1) |
| `weights` | the KL divergence from the normalised `weights` is at most `max_kl_bits` | `weights` (outcome → weight), `max_kl_bits` |
| `answer` | the expected token's probability is at least `min_p` | `value`, `min_p` (default 0.99) |
| `min_entropy` | the read's entropy is at least `bits` | `bits` |

Outcome masses come from the read's `tracked` (each outcome by its own
name), else from its `top` by exact token text. A read with no mass on any
outcome is reported as *unjudgeable* (`pass: null` with a `note`) rather
than counted as a failure — a hole in the read must not masquerade as a
verdict.

The header's `summary` carries the pass rate: the number a write-up cites.
""",
    inputs=(
        In("results", "logits/distribution",
           "The decision reads — a collection of `logits/decision`, or of any "
           "kind that extends `logits/distribution`.", many=True),
        In("expectations", "records/record",
           "Records `{id, expect}`, one per result to judge.", many=True),
    ),
    emits=(
        Emits('eval/verdict', collection=True, doc='One verdict per judged result: `id`, `coords`, `expect`, `entropy_bits`, `kl_bits`, `mass`, `p_expected`, and `pass` (null with a `note` when unjudgeable). The header\'s `summary` carries `pass_rate`, `n_pass`, `n_judged` and `n_unjudgeable`.')
    ),
    params=(),
    example={},
    example_inputs={
        "results": {"$fetch": "$reads"},
        "expectations": [
            {"id": "d6", "expect": {"type": "uniform",
                                    "over": ["1", "2", "3", "4", "5", "6"],
                                    "max_kl_bits": 0.1}},
        ],
    },
)

VIZ_SPEC = Op(
    name="records/chart",
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
        In("records", "records/record | records/table",
           "The table or records to chart.", many=True),
    ),
    emits=Emits('records/chart', collection=False, doc='`title`, `mark`, `encoding` (`x`, `y`, `series`), and `source` or `data`.'),
    params=(
        P("encoding", "object",
          "`{\"x\": field, \"y\": field, \"series\": field}` — which fields "
          "go on which axis, and which splits the data into series.",
          None),
        P("x", "string", "The x field; the same as `encoding.x`.", None),
        P("y", "string", "The y field; the same as `encoding.y`.", None),
        P("mark", "string", "The mark: `\"bar\"`, `\"line\"`, `\"point\"`.", "bar"),
        P("title", "string", "The chart's title.", ""),
    ),
    example={
        "title": "Δ log p by layer",
        "mark": "line",
        "encoding": {"x": "layer", "y": "mean", "series": "genre"},
    },
    example_inputs={"records": {"$fetch": "$table"}},
)

VECTORS_SIMILARITY = Op(
    name="geometry/similarity",
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
and emits the matrix with the metric, its `options`, and whether it is
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
with `options: {"center": true}`, which subtracts it, and `geometry/mst`
downstream.
""",
    inputs=(
        In("items", "activations/vector | logits/distribution | records/record",
           "The items to compare: any collection whose kind declares metrics.",
           many=True),
    ),
    emits=(
        Emits('geometry/similarity', collection=True, doc='One item per group: `{group, layer?, head?, space?, ids, labels, matrix, pairs?, separation?, nn_purity?, silhouette?}`, `labels` being the items\' values on the `axis` coordinate. The header carries `metric`, `metric_kind`, `symmetric`, `options`, `over` (the kind compared), `by`, `axis`, and `position`/`point` for vectors.')
    ),
    params=(
        P("metric", "string",
          "Which of the kind's metrics to apply. By default the kind's first: "
          "`cosine` for vectors, `jensen-shannon` for distributions, "
          "`hamming` for records.",
          None),
        P("options", "object",
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
    example_inputs={"items": {"$fetch": "$vectors"}},
)

VECTORS_MST = Op(
    name="geometry/mst",
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
           "The pairwise matrices, one per group, from `geometry/similarity`.",
           many=True),
    ),
    emits=Emits('geometry/mst', collection=True, doc='One item per group: `group`, `layer`/`head` when the group is a space, `n`, `n_edges`, `mean`, `variance`, `stdev`, `cv`, `total`, `min`, `max`, `bridge_threshold`, `bridges`, `components_after_cut`, `ids`, `labels`, and `edges` as `[i, j, weight]` when kept. The header carries `metric`, `options`, `over` (the kind compared), `bridge_sigma` and `axis`. `records/table` reads the items as its rows.'),
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
    example_inputs={"similarity": {"$fetch": "$similarity"}},
)

OPS: tuple[Op, ...] = (
    FACTOR_CROSS, TEMPLATE, RENAME, SELECT, UNION, PAIRED_DELTA, GROUP_STATS,
    TABLE_FROM_RECORDS, TEXT_STATS, REDUCE_SUM, REDUCE_TOP_K, REDUCE_HISTOGRAM,
    EVAL_EXPECTATION, VIZ_SPEC, VECTORS_SIMILARITY, VECTORS_MST,
)
