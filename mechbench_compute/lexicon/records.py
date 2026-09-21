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

`over` is the other way to give the stream: a list of values, or
`{"range": [0, 42]}`, each becoming a record of one field named by `as`.
`{"over": {"range": [0, 42]}, "as": "layer"}` runs the body once per
layer with `{"$param": "layer"}` filled in — where before the integers
had to be stored as a corpus first.

`collect` says what comes back. `stream` (the default) flattens every
invocation's output into one collection, each item's id prefixed with
its record's and carrying a `mapped` coordinate; `first` keeps one item
per record; `all` keeps each invocation's items nested under its record.
When the body ends at more than one node, `output` names the one to
collect.
""",
    inputs=(
        In("records", "records/record",
           "The stream to map over: one invocation of the body per record. "
           "`over` is the other way to give one.",
           many=True, required=False),
    ),
    output=Output('records/record', collection=True, doc="Under `stream`, every invocation's items in one collection, each id prefixed `<record>:<item>` and carrying `coords.mapped`; under `first` or `all`, one item per record. Under `stream` and `first` the collection takes the BODY's item kind — a map over transcripts that produces transcripts emits transcripts — and under `all`, where each item nests a list, it is a plain record. The header's `mapped` says how many records ran, under which policy, and what the body's nodes were."),
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
        P("over", "json",
          "The values to map over, in place of the `records` port: a list, "
          "or `{\"range\": [0, 42]}`. Each becomes a one-field record, so a "
          "sweep over layers needs no corpus of integers stored first.",
          None),
        P("as", "string",
          "What `over`'s value is called inside the body — the `$param` it "
          "fills, and the coordinate it lands on.",
          "value"),
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

FOLD = Op(
    name="records/fold",
    summary=(
        "Run a body graph step after step, each step reading the state the "
        "last one wrote — the loop a conversation, a refinement or an "
        "agentic round is made of — until the steps run out or the state "
        "says stop."
    ),
    description="""\
`records/map` runs a body once per record with no memory between runs;
this runs it once per **step**, and what the body produces at step *t* is
what it reads at step *t + 1*. The state arrives on the `state` port,
enters the body on the body input named `state` (an edge `{"from":
{"input": "state"}}`), and comes back out of the body's output named by
`output` (or its one output). The final state is the result.

**Steps.** `over` is a list of objects, one per step, cycled when `steps`
is longer than it: each object's keys are `$param`s the body's nodes read
that step — `[{"participant": "ana"}, {"participant": "bo"}]` with `steps:
6` is a six-turn round robin. A body that needs no per-step values takes
`steps` alone. The step's index is bound as `$step`.

**Stopping.** `until: {"field": "stopped"}` ends the fold early when every
item of the state has a non-empty value in that field — which is how a
body says the conversation reached its stop phrase (`text/extend` writes
`stopped`), the answer converged, or the tool loop finished. The header's
`folded` says how many steps ran and why it ended.

Every step is an item keyed by its index, so an interrupted fold resumes
at the step it reached with the state it had. A body sees one state at a
time and nothing else.

The conversation: `text/render` → `text/chat` → `text/extend` as the
body, transcripts as the state, participants as `over` — a multi-party
conversation is three ops and a loop, not an operation of its own.
""",
    inputs=(
        In("state", "collection",
           "The starting state: what the body reads at step 0 — any "
           "collection, of whatever kind the body's `state` input takes.",
           many=True),
    ),
    output=Output('records/record', collection=True, doc="The state after the last step — the body's `output` at that step, its kind whatever the body's output node emits — with `folded` on the header: `steps` (how many ran), `stopped` (`\"steps\"`, or `\"until\"` when the state said stop), `body_nodes`."),
    params=(
        P("body", "object",
          "The graph to run per step — `{nodes, edges}`, the same shape a "
          "protocol's graph has. An edge from `{\"input\": \"state\"}` "
          "carries the state in; `output` names the node that carries it out.",
          fields=(
              P("nodes", "list[json]", "The body's nodes, as a protocol graph writes them."),
              P("edges", "list[json]", "The body's edges.", []),
          )),
        P("over", "list[json]",
          "One object per step, its keys the `$param`s the body reads that "
          "step — open by design, since they are the body's names; cycled "
          "when `steps` exceeds its length.",
          None),
        P("steps", "int",
          "How many steps to run. Defaults to the length of `over`.",
          None),
        P("until", "object",
          "Stop early when every state item has a non-empty value in "
          "`field`.",
          None, fields=(P("field", "string", "The state field that says stop."),)),
        P("output", "string",
          "Which of the body's terminal nodes carries the state out, when "
          "it has more than one.",
          None),
    ),
    example={"over": [{"participant": "ana"}, {"participant": "bo"}], "steps": 6,
             "until": {"field": "stopped"}, "output": "next",
             "body": {"nodes": [
                 {"id": "view", "block": "text/render", "params": {"participant": {"$param": "participant"}}},
                 {"id": "say", "block": "text/chat", "params": {"model": {"$param": "model"}}},
                 {"id": "next", "block": "text/extend", "params": {"participant": {"$param": "participant"}}}],
                      "edges": [
                 {"from": {"input": "state"}, "to": {"node": "view", "port": "transcripts"}},
                 {"from": {"node": "view"}, "to": {"node": "say", "port": "records"}},
                 {"from": {"input": "state"}, "to": {"node": "next", "port": "transcripts"}},
                 {"from": {"node": "say"}, "to": {"node": "next", "port": "replies"}}]}},
    example_inputs={"state": {"$ref": {"bench": "you/lab/openings"}}},
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

A grid — a patch trace, a head sweep, a lens read-out, anything with
`axes` and `measures` — is summarised cell by cell: each cell is a record
with the axes as fields (and `token` beside `position` when the grid
carries tokens) and each measure a column. `value: "share", by:
["position"]` over a trace is one row per token with the most any layer's
patch there recovers.

A record without the value field is refused by name, because a mean over
"the records that happened to have it" is the kind of number nobody
notices is wrong. When absent values are expected — a judge that could not
be read, an unscored item — set `on_missing: "skip"` and the count of
skipped records is reported on the table as `n_missing`.

### How sure

A mean over fifteen prompts is a point; `interval: 0.95` puts an interval
around it — `lo` and `hi`, the percentile bootstrap of the mean over
`resamples` redraws of the group's records under `seed` — so a peak in a
sweep is a claim with a width, not a number. Whether two groups DIFFER is
`records/contrast`'s question, which pairs the records first.
""",
    inputs=(_RECORDS,),
    output=(
        Output('records/table', collection=False, doc='One row per group with the `by` coordinates and `n`, `median`, `mean`, `min`, `max`, `share_negative`, plus `lo` and `hi` when an `interval` was asked; `n_missing` when any were skipped; the header\'s `interval` says the level, method, resamples and seed.')
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
        P("interval", "float",
          "The level of a bootstrap interval on each group's mean — `0.95` "
          "adds `lo` and `hi` to every row. None reports the point alone.",
          None),
        P("resamples", "int",
          "How many bootstrap redraws the interval is read from.",
          2000),
    ),
    example={"value": "delta", "by": ["genre", "alpha"], "interval": 0.95},
    example_inputs={"records": {"$ref": {"bench": "you/lab/deltas"}}},
)

CONTRAST = Op(
    name="records/contrast",
    summary=(
        "The difference between two conditions' means of a field — a minus "
        "b on one coordinate — with a paired bootstrap interval and how "
        "often the sign held, one row per combination of the other "
        "coordinates."
    ),
    description="""\
`records/summarize` says what each condition's mean is; this says whether
two of them differ. Name the coordinate (`on`) and its two values (`a`,
`b`); every record on either side contributes its `value`, and the row
reports `mean_a`, `mean_b`, their difference `diff`, the interval `lo`–`hi`
at the given level, and `share_positive` — the fraction of bootstrap
redraws in which the difference was above zero, which is how often the
sign held.

**Pair the records.** In a sweep the same prompts run under every
condition, and a prompt that is hard is hard on both sides. `paired`
names the field the two sides share — a prompt's `id`, usually — and the
bootstrap then redraws PAIRS, so each record's own noise cancels the way
it does in the data; only records present on both sides count, and `n`
is their number. Without `paired`, the sides are redrawn independently,
which is right when the two conditions ran on different records and
wider than it needs to be when they did not.

`by` names further coordinates to hold fixed: one row per combination
of them, each its own contrast.
""",
    inputs=(_RECORDS,),
    output=(
        Output('records/table', collection=False, doc='One row per combination of the `by` coordinates: `on`, `a`, `b`, `n`, `mean_a`, `mean_b`, `diff`, `lo`, `hi`, `share_positive`. The header\'s `interval` says the level, method, whether it was paired, resamples and seed.')
    ),
    params=(
        P("value", "string", "The numeric field compared."),
        P("on", "string", "The coordinate (or top-level field) the two conditions differ on."),
        P("a", "json", "The value of `on` on the first side; the difference is a minus b."),
        P("b", "json", "The value of `on` on the second side."),
        P("paired", "string",
          "The field the two sides share, to redraw pairs — `\"id\"` when "
          "the same records ran under both conditions. None redraws the "
          "sides independently.",
          None),
        P("by", "list[string]",
          "Coordinates to hold fixed: one row per combination.",
          None),
        P("interval", "float", "The level of the bootstrap interval.", 0.95),
        P("resamples", "int", "How many bootstrap redraws the interval is read from.", 2000),
    ),
    example={"value": "delta_logp", "on": "layer", "a": 23, "b": 22, "paired": "id"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/ablation"}}},
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

### The two marks interp actually needs

Half of this platform's figures are grids: (layer, position) from
`intervene/patch`, (layer, head) from `intervene/ablate-heads` or
`intervene/path`, (layer, prompt) from `logits/attribute`. That is
`mark: "heat"` with `encoding: {x, y, value}` — one cell per row, the
value its colour, `scale: "diverging"` centred on zero where the number
is a change and `"sequential"` where it is a magnitude.

The other half are token strips: a prompt's own tokens, each coloured by
a number it carries — a per-token surprisal from
`activations/capture-tokens`, a probe's projection, the window
`activations/examples` brings back. That is `mark: "tokens"` with
`encoding: {text, value}`, where `text` names the field holding the
row's tokens. Rows that are one token each — a grid's cells, or a
`records/summarize` over them by position — make a strip too: `text`
names the token field, `x` the position (default `position`), and
`facet` (or `series`) says which rows are one prompt, one strip per
value on one colour scale.

An `lo`/`hi` encoding draws the interval `records/summarize` reports
beside the point it belongs to.

### What makes it a visualization

A figure on this platform is meant to lend a reader spatial intuitions
for a space they have none for, and it carries four things beyond the
mark to do it (the vocabulary is `mechbench/docs/VISUALIZATION.md`):

* **`labels`** — what each field is called in prose. The axes are
  labelled with these words and the hover readout is a sentence built
  from them: "Layer 23, attention only: −3.10 in log probability".
* **`axes.layer`** — the model's depth landmarks: how many layers,
  which attend globally, and where fresh keys and values stop. With
  them, the depth axis marks the global-attention layers and the
  key/value boundary, so a layer is the same place in every figure. A
  sweep's result carries them under `arch`, and a summary or contrast
  of it carries them forward; this op reads them from its input when
  `axes` is not given.
* **`annotate`** — callouts drawn on the figure at named rows. The
  extremes are labelled by default; this names what else to say.
* **`focus`** — the field this figure shares with the others on a page:
  hover a layer here and it lights on every figure that has one.
  Defaults to `layer` when the rows carry it, else `x`.
* **`facet`** — small multiples: one panel per value of the field, on
  one shared x axis, each with its own value scale. A sweep whose
  largest series would flatten the others is four panels, not four
  lines.

`encoding.color` tints each mark by a categorical field **in place**;
`encoding.series` splits rows into several lines or side-by-side bars.
The two are different and may be combined.
""",
    inputs=(
        In("records", "collection | records/table",
           "The table, or any collection of items, to chart.", many=True),
    ),
    output=Output('records/chart', collection=False, doc='`title`, `mark`, `encoding` (`x`, `y`, `series`, `color`, `value`, `text`, `lo`, `hi` as the mark uses them), `scale` when given, `labels`, `axes`, `annotate`, `focus` and `facet` when given, and `source` or `data`.'),
    params=(
        P("encoding", "object",
          "Which field goes where. `x`/`y` for the point-shaped marks, "
          "`series` to split into series, `value` for a heat cell or a "
          "token's colour, `text` for the token strip's tokens, `lo`/`hi` "
          "for an interval.",
          None, fields=(
              P("x", "string", "The field on the x axis.", None),
              P("y", "string", "The field on the y axis.", None),
              P("series", "string", "The field that splits the rows into series.", None),
              P("color", "string",
                "The categorical field each mark takes its colour from, in place.", None),
              P("value", "string", "The number a heat cell or a token takes its colour from.", None),
              P("text", "string", "The field holding a row's tokens, for a token strip.", None),
              P("lo", "string", "The interval's lower end.", None),
              P("hi", "string", "The interval's upper end.", None),
          )),
        P("x", "string", "The x field; the same as `encoding.x`.", None),
        P("y", "string", "The y field; the same as `encoding.y`.", None),
        P("value", "string", "The value field; the same as `encoding.value`.", None),
        P("text", "string", "The tokens field; the same as `encoding.text`.", None),
        P("lo", "string", "The lower end; the same as `encoding.lo`.", None),
        P("hi", "string", "The upper end; the same as `encoding.hi`.", None),
        P("mark", "string",
          "`\"bar\"`, `\"line\"`, `\"point\"`, `\"heat\"` (a grid of cells) "
          "or `\"tokens\"` (a prompt's tokens, coloured).",
          "bar", choices=("bar", "line", "point", "heat", "tokens")),
        P("scale", "string",
          "How `value` becomes colour: `\"diverging\"` centres on zero (a "
          "recovery, a Δ log p), `\"sequential\"` runs from the lowest "
          "value. Diverging when the values cross zero, by default.",
          None, choices=("diverging", "sequential")),
        P("title", "string", "The chart's title: its claim, as a sentence.", ""),
        P("labels", "object",
          "What each encoded field is called in prose — the axis labels "
          "and the words the hover readout uses.",
          None, fields=(
              P("x", "string", "What the x field is called.", None),
              P("y", "string", "What the y field is called.", None),
              P("value", "string", "What the value field is called.", None),
              P("series", "string", "What the series field is called.", None),
              P("color", "string", "What the colour field is called.", None),
          )),
        P("axes", "object",
          "Landmarks for an axis. `layer` carries the model's depth: "
          "`n` layers, the `global` attention layers, and "
          "`kv_shared_from`, the first layer that reuses keys and values. "
          "Read from the input's `arch` header when not given.",
          None, fields=(
              P("layer", "object", "The depth landmarks.", None, fields=(
                  P("n", "int", "How many layers the model has.", None),
                  P("global", "list[int]", "The layers that attend to the whole prompt.", None),
                  P("kv_shared_from", "int",
                    "The first layer that reuses earlier keys and values.", None),
              )),
          )),
        P("annotate", "list[object]",
          "Callouts drawn on the figure: each names the row it sits at "
          "(`at`, a field → value map) and what it says.",
          None, fields=(
              P("at", "map[string, json]", "The row: field → value.", None),
              P("text", "string", "What the callout says.", None),
          )),
        P("reference", "list[object]",
          "Lines the marks are read against: a target, a baseline, a "
          "threshold, chance. Each names a value on one axis — `y` for a "
          "rule across the plot, `x` for one down it — and what it is. "
          "A figure whose claim is \"close to fair\" or \"above the "
          "threshold\" cannot make it without one.",
          None, fields=(
              P("y", "float", "The value on the y axis to rule at.", None),
              P("x", "json", "The value on the x axis to rule at.", None),
              P("text", "string", "What the line is: `a fair die`, `chance`.", None),
          )),
        P("focus", "string",
          "The field shared with the other figures on a page, so a hover "
          "here lights the same value there. `layer` when the rows carry "
          "one, else the x field.",
          None),
        P("facet", "string",
          "Small multiples: one panel per value of this field, stacked on "
          "one shared x axis, each with its own value scale — four sweeps "
          "as four aligned panels rather than four lines on one scale.",
          None),
    ),
    example={
        "title": "Removing only the attention at each layer",
        "mark": "point",
        "encoding": {"x": "layer", "y": "mean", "lo": "lo", "hi": "hi",
                     "color": "attention"},
        "labels": {"y": "change in the answer's log probability",
                   "color": "attention kind"},
        "annotate": [{"at": {"layer": 23}, "text": "the last global layer with fresh keys and values"}],
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
    CONTRAST,
    FOLD,
    TABLE_FROM_RECORDS, TEXT_STATS, REDUCE_SUM, REDUCE_TOP_K, REDUCE_HISTOGRAM,
    EVAL_EXPECTATION, VIZ_SPEC, VECTORS_SIMILARITY, VECTORS_MST,
)
