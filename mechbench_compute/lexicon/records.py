"""Ops over **records**: making them, filtering, joining, summarising and
presenting them. None of these touch a model; they are the plumbing
between the ops that do.

A record is a JSON object with an `id`, usually a `coords` object (the
experimental condition it belongs to — `{"genre": "noir", "seed": 3}`),
and whatever fields the op that made it wrote. Most of the ops here read
records from an edge onto their `records` port.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Emits, Op, P

_RECORDS_IN = "`records` (by edge) — the records to work on."

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
    inputs="None — this op makes records from its params.",
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

The output records keep their `id` and `coords`, so a downstream
`logits/decision` or `text/generate` can be told `user_field: "question"` and read
the field this op wrote.
""",
    inputs="`records` (by edge, or the `records` param) — records with `values`, usually from `records/cross`.",
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
    inputs=_RECORDS_IN,
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
    inputs="Any number of edges, each carrying a collection (of records, or of `activations/vector`). Port names are the values on the batch coordinate.",
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
    inputs=_RECORDS_IN,
    emits=Emits('records/record', collection=True, doc='One record per non-baseline record: `{id, coords, value, baseline, delta}`.'),
    params=(
        P("value", "string", "The numeric field to difference."),
        P("baseline_where", "object",
          "Coordinates identifying the baseline records, e.g. "
          "`{\"alpha\": 0}`."),
        P("match_on", "list[string]",
          "Coordinates a record and its baseline must agree on. Empty "
          "means one baseline for everything.",
          None),
    ),
    example={"value": "entropy_bits", "baseline_where": {"alpha": 0}, "match_on": ["prompt"]},
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
    inputs=_RECORDS_IN,
    emits=(
        Emits('records/table', collection=False, doc='One row per group with the `by` coordinates and `n`, `median`, `mean`, `min`, `max`, `share_negative`; `n_missing` when any were skipped.')
    ),
    params=(
        P("value", "string", "The numeric field to summarise."),
        P("by", "list[string]",
          "The coordinates to group on. Empty gives one overall row.",
          None),
        P("on_missing", "string",
          "`\"error\"`: refuse a record without the field. `\"skip\"`: omit "
          "it and report how many were omitted.",
          "error"),
    ),
    example={"value": "delta", "by": ["genre", "alpha"]},
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
    inputs="`records` (by edge, or the `records` param).",
    emits=Emits('records/table', collection=False, doc='`columns` (`{name, dtype}`) and `rows`.'),
    params=(
        P("row_axis", "string",
          "What one row stands for, recorded on the table for its renderer "
          "— `\"record\"`, `\"condition\"`, `\"layer\"`.",
          "record"),
    ),
    example={"row_axis": "condition", "name": "steering deltas"},
)

TEXT_STATS = Op(
    name="text/stats",
    summary=(
        "Measure each text in a corpus — pattern hits, word and distinct-word "
        "counts, vocabulary rarity against a frequency table — and either "
        "annotate the records or summarise the corpus."
    ),
    description="""\
Each entry of `measures` is applied to every record's `field`:

| `type` | Fields written per record | Options |
|---|---|---|
| `pattern` | `<name>`: 1 if any regex matches, else 0 | `patterns` (list of regexes), `where`: `"anywhere"` or `"prefix"` (must match at the start), `ignore_case` |
| `lexical` | `<name>_words`, `<name>_distinct`, `<name>_dup` (1 − distinct/words) | `lowercase` (default true), `min_length` |
| `corpus_frequency` | `<name>`: the statistic over the reference frequency of the text's words; `<name>_coverage`: the fraction of words found in the table | `frequencies` (word → count, or wire a `frequencies` input), `stat`: `"mean_log10"` (rarer vocabulary ⇒ lower), `"mean"` or `"coverage"`, `lowercase`, `min_length` |

In `annotate` mode the output is the records with those fields added —
ready for `records/select`, `records/stats` or `trajectory/capture` (which can label
by them). In `corpus` mode it is one summary record: per pattern a count and
rate, corpus-wide word and distinct-word counts and duplication, and the
mean of each frequency statistic.
""",
    inputs=(
        "`records` or `documents` (by edge, or the `records` param) — a "
        "record list or a document collection. `frequencies` (optional, by "
        "edge) — a word-frequency table for `corpus_frequency` measures."
    ),
    emits=(
        Emits('records/record', collection=True, doc='In `annotate` mode, one record per item (`id`, `coords`, the measure fields, and the whole item when `keep` is set). In `corpus` mode, a single record with the corpus summary.')
    ),
    params=(
        P("measures", "list[object]",
          "The measurements to make, each `{kind, name, …}` as in the "
          "table above.",
          None),
        P("field", "string", "The record field holding the text.", "text"),
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
        "field": "text",
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
    inputs=_RECORDS_IN,
    emits=Emits('records/sum', collection=False, doc='`{n, sum}`.'),
    params=(P("value", "string", "The numeric field to sum."),),
    example={"value": "cost_usd"},
)

REDUCE_TOP_K = Op(
    name="records/top-k",
    summary="The k records with the largest value of a field.",
    description="""\
Sorted by the field descending, ties broken by `id`, so the result is
deterministic. An exact reduce: the top-k of a union is the top-k of the
top-ks, so partial results merge without loss.
""",
    inputs=_RECORDS_IN,
    emits=Emits('records/record', collection=True, doc='The top k, in order.'),
    params=(
        P("value", "string", "The numeric field to rank by."),
        P("k", "int", "How many to keep.", 10),
    ),
    example={"value": "delta", "k": 5},
)

REDUCE_HISTOGRAM = Op(
    name="records/histogram",
    summary="Count a numeric field into fixed, equal-width bins.",
    description="""\
`bins` equal-width bins span `[lo, hi)`; values below `lo` and at or above
`hi` are counted separately rather than dropped, so the total always equals
the number of records. An exact reduce: counts add.
""",
    inputs=_RECORDS_IN,
    emits=Emits('records/histogram', collection=False, doc='`bins` (the counts, in order), `below`, `above`.'),
    params=(
        P("value", "string", "The numeric field to bin."),
        P("lo", "float", "The lower edge of the first bin."),
        P("hi", "float", "The upper edge of the last bin (exclusive)."),
        P("bins", "int", "How many equal-width bins between `lo` and `hi`."),
    ),
    example={"value": "entropy_bits", "lo": 0.0, "hi": 8.0, "bins": 16},
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
        "`results` (by edge or param) — a collection of `logits/decision` "
        "(or of any kind that extends `logits/distribution`). `expectations` "
        "(by edge or param) — records `{id, expect}`."
    ),
    emits=(
        Emits('eval/verdict', collection=True, doc='One verdict per judged result: `id`, `coords`, `expect`, `entropy_bits`, `kl_bits`, `mass`, `p_expected`, and `pass` (null with a `note` when unjudgeable). The header\'s `summary` carries `pass_rate`, `n_pass`, `n_judged` and `n_unjudgeable`.')
    ),
    params=(
        P("results", "record | list[record]",
          "The decision reads, when they do not arrive by edge.",
          None),
        P("expectations", "list[record]",
          "The expectations, when they do not arrive by edge.",
          None),
    ),
    example={
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
    inputs="`records` (by edge) — a table or record list to chart.",
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
)

VECTORS_SIMILARITY = Op(
    name="geometry/similarity",
    summary=(
        "The pairwise cosine similarity of residual vectors at each layer, "
        "with how well the labels separate — intra- vs inter-label "
        "similarity, nearest-neighbour purity, silhouette."
    ),
    description="""\
For each layer (and head, for Q/K vectors) in the collection, the full
cosine matrix over its items. When every item has a value on the `axis`
coordinate and there is more than one value, the block also reports the mean
cosine within groups, between groups, their gap, the fraction of items whose
nearest neighbour shares their group, and the silhouette score.

Raw cosine between transformer activations is dominated by a shared
direction they all lean toward; for a variety measure prefer `geometry/mst`
with `center: true`, which subtracts it.
""",
    inputs="`vectors` (by edge, or the `vectors` param) — a collection of `activations/vector`.",
    emits=(
        Emits('geometry/similarity', collection=True, doc='One item per group: `{layer, head?, space, ids, labels, matrix, separation?, nn_purity?, silhouette?}`, `labels` being the items\' values on the `axis` coordinate. The header carries `position`, `point`, `metric` and `axis`.')
    ),
    params=(
        P("axis", "string",
          "The coordinate the separation metrics group on. `label` reads the "
          "older `label` field as well.",
          "label"),
    ),
    example={"axis": "genre"},
)

VECTORS_MST = Op(
    name="geometry/mst",
    summary=(
        "Measure how varied a set of vectors is with a minimum spanning tree "
        "over their distances — the spread, its clumpiness, and how many "
        "clusters the bridges imply."
    ),
    description="""\
Average pairwise distance cannot tell three tight clumps from an even
spread; a minimum spanning tree can. Its edges are the cheapest set that
still connects every point, so within a clump edges are short and between
clumps there is one long **bridge**. Per layer the block reports the mean
edge (the scale of the spread — small means collapsed), the variance and
coefficient of variation (clumpiness), and the number of bridges — edges
more than `bridge_sigma` standard deviations above the mean — which is
roughly the cluster count minus one. Read `mean` and `variance` together:
a collapsed corpus and an evenly varied one both have low variance, for
opposite reasons.

`center: true` subtracts the mean vector before measuring. Transformer
activations occupy a narrow cone around one dominant direction, and raw
cosine between two of them mostly measures that shared direction; centering
removes it, and the rankings it produces agree across layers and pooling
choices where the uncentered ones do not. It is off by default only so
stored results keep their numbers; new protocols should turn it on. It
needs vectors, not a similarity matrix.

The tree is built deterministically (ties break toward the lower index), so
a run that reproduces its numbers reproduces its tree.
""",
    inputs=(
        "`vectors` (by edge or param) — a collection of `activations/vector`; "
        "or `matrix` / `similarity` (by edge, or the `matrix` param) — a "
        "collection of `geometry/similarity`."
    ),
    emits=Emits('geometry/mst', collection=True, doc='One item per group (a layer, or a layer and head): `n`, `n_edges`, `mean`, `variance`, `stdev`, `cv`, `total`, `min`, `max`, `bridge_threshold`, `bridges`, `components_after_cut`, `ids`, `labels` (the items\' values on the `axis` coordinate), and `edges` as `[i, j, weight]` when kept. The header carries `metric`, `centered`, `bridge_sigma` and `axis`. `records/table` reads the items as its rows.'),
    params=(
        P("axis", "string",
          "The coordinate reported as each item's label. `label` reads the "
          "older `label` field as well.",
          "label"),
        P("center", "bool",
          "Subtract the mean vector before measuring distance. Recommended "
          "on; requires `vectors` rather than a similarity matrix.",
          False),
        P("bridge_sigma", "float",
          "How many standard deviations above the mean edge an edge must be "
          "to count as a bridge between clusters.",
          2.0),
        P("keep_edges", "bool",
          "Include every tree edge (`[i, j, weight]`) in the output, not "
          "only the statistics.",
          True),
    ),
    example={"vectors": {"$fetch": "$vectors"}, "center": True},
)

OPS: tuple[Op, ...] = (
    FACTOR_CROSS, TEMPLATE, SELECT, UNION, PAIRED_DELTA, GROUP_STATS,
    TABLE_FROM_RECORDS, TEXT_STATS, REDUCE_SUM, REDUCE_TOP_K, REDUCE_HISTOGRAM,
    EVAL_EXPECTATION, VIZ_SPEC, VECTORS_SIMILARITY, VECTORS_MST,
)
