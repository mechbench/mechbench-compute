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
    
    
    
    
    TEXT_STATS, 
    EVAL_EXPECTATION, VECTORS_SIMILARITY, VECTORS_MST,
)
