"""The families: the namespaces the operations and the kinds are grouped
in, described for the reader choosing where to look.

A family names the thing its members read or produce, not a technique
(docs/LEXICON.md §2). Every operation and every kind belongs to exactly
one; the eleven families with operations have kinds too, and four
platform families have kinds only. The prose here is what the family
page on the documentation site says above its members.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Family

FAMILIES: tuple[Family, ...] = (
    Family(
        "records",
        "The rows of an experiment: the conditions a design produces, the results every operation produces, and the tables that summarise them.",
        """\
Everything an operation reads or produces is a record — an `id`, the
`coords` that place it in the design, and the fields the operation
wrote — and everything plural is a collection of records of one kind.
This family makes records (`cross` writes one per combination of
factors, `fill` turns them into prompts), reshapes them (`select`,
`rename`, `union`, `subtract`), reduces them (`summarize`, `total`,
`rank`, `bin`), compares two collections of them record by record
(`diff`, the question a re-run asks of its predecessor) and presents
them (`tabulate`, `plot`).

Two rules run through it. A collection grows by union and never by
mutation, so a node's output is a new object and the input is untouched.
And an operation reads a typed kind, never a field a parameter names: a
record whose text sits under the wrong name is adapted by a `rename`
node that the graph shows, not by telling every consumer where to look.
""",
    ),
    Family(
        "text",
        "Generating text from a model and measuring what came out: documents, transcripts, their tokens and their statistics.",
        """\
The generating operations sample from local weights or call a hosted
model through the same node (`generate`, `chat`); each writes a document
with the sampling and the cost recorded per item. A conversation is
those same nodes folded over turns: `render` shows one participant the
room, `chat` asks it, `extend` appends what it said. The
measuring operations read those back: `score` annotates every token
with its surprisal under a model, `measure` counts patterns and
vocabulary across a corpus, and `tokenize` measures how a tokenizer
splits a set of items — the depth and fragmentation that decide whether
a decision read can be taken at one token.
""",
    ),
    Family(
        "eval",
        "Judgments on records: a model as grader, an expectation checked on a decision read, a standard metric, a benchmark suite.",
        """\
An evaluation turns a record into a verdict. `judge` has a model grade
each record against a rubric — a score, a label or an A/B preference,
with repeated votes and the position order randomised and recorded.
`expect` checks a decision read against a claim carried as data:
uniform over the outcomes, a required answer, a minimum entropy, a
target distribution. `score` scores prediction against reference with
a named standard metric, and `benchmark` runs benchmark tasks against
the model as the platform loads it, so pinned revisions and fused
adapters count. All four emit `eval/verdict`, one per record, with the
rate or the summary on the collection's header.

A judge is not a metric in the geometry sense: its preferences are not
symmetric, and a verdict is a reading of one record, not a distance
between two.
""",
    ),
    Family(
        "logits",
        "The model's next-token distribution: read at the decision point, at every layer, at every position, or split by what contributed.",
        """\
Every operation here reads the model's output distribution and produces a
`logits/distribution` or a kind that extends it — the same `entropy_bits`,
`top` and `tracked` spelled once, so a decision read, a funnel layer and
an intervention readout compare by the same metrics. `read` reads at
the decision point, the first token after a prompt and its prefill, and
produces a decision. `read-layers` reads that point at every layer through
the unembedding — the funnel, the curve of a model committing. `scan`
reads every position at every layer for one target token — the lens.
`attribute` splits the final logit into the contribution of the
embedding and of each layer, with the check that the pieces sum to the
whole.
""",
    ),
    Family(
        "activations",
        "What the forward pass computes: residual vectors, the drift between two prompts' streams, and the attention patterns.",
        """\
`capture` records the residual stream at chosen layers and a chosen
position, or pooled over positions — the raw material of every
geometry measurement and every direction. `contrast` runs a matched
pair and measures at every (layer, position) how far their streams have
moved apart — the divergence. `capture-attention` records each head's
weights over the prompt.

The kinds are the vector and the grid. A vector carries its `space` and
is the ancestor of a direction and of a trajectory point; a coordinate is
a vector's scalar along a direction. A grid is a scalar field over model
axes — layer, position, head — and is the ancestor of every map an
operation draws.
""",
    ),
    Family(
        "geometry",
        "Comparing items under a metric their kind declares: the pairwise matrix, and the spanning tree that measures variety.",
        """\
A kind declares how its items compare the way it declares how they are
drawn: vectors by cosine, euclidean distance or dot product; next-token
distributions by Jensen–Shannon, Hellinger, total variation or KL;
records by how many coordinate axes differ. `compare` takes any such
collection and produces the similarity matrix per group, with the metric
and its options recorded and, when the items carry a value on the
chosen axis, how well the groups separate. `span` builds a minimum
spanning tree over a similarity — the spread's scale, its clumpiness,
and the bridges that imply clusters — for whatever the similarity
compared.

So the same two nodes answer "are these adapters' axes aligned", "do
these conditions make the model say the same thing", and "how varied is
this corpus": the kind chooses the metric, the geometry is the same.
""",
    ),
    Family(
        "intervene",
        "Editing the forward pass and reading what changed: ablate a layer or a head, add a direction, patch one run into another.",
        """\
`apply` is the general operation: edit activations at chosen points —
zero them, scale them, add or remove a direction, patch them from another
run — and read out the next-token distribution or the activations that
result, over a sweep of strengths with a control. The others are its
common cases, each with its own readout: `ablate-layers` removes one
layer's contribution at a time, `ablate-heads` one head at a time,
`steer` adds a direction built from labelled vectors and sweeps its
strength, and `patch` patches a clean run into a corrupted one at each
(layer, position) to map where the answer comes back — the causal
trace.

Every readout names the target token whose log-probability is followed —
by the `tracked` parameter, or the model's own top prediction when
nothing is named — and reports the untouched baseline it is measured
against.
""",
    ),
    Family(
        "direction",
        "Directions in a model's activation space as objects: made from vectors, combined, and read back through the vocabulary.",
        """\
A direction is a unit vector in one activation space with its derivation
attached — how it was made, from what, on which model. `fit` makes one
as the difference between two groups' centroids, the axis along which
one condition differs from another; `decompose` as the principal
component along which a set varies most. `add`, `average`,
`orthogonalize` and `normalize` combine directions in the same space,
each carrying the derivation forward. `project` reads a vector's
coordinate along a direction; `unembed` reads the direction through the
unembedding, the tokens it promotes and the tokens its negative
promotes.

A direction is a vector, so the geometry operations compare directions
as they compare any vectors: a union of several is a collection, and one
similarity over it is the matrix of their cosines.
""",
    ),
    Family(
        "trajectory",
        "The residual stream followed along an axis: a prompt through every layer, or a text along every position.",
        """\
`capture` follows the residual stream step by step — one position through
the layers, or one layer along the positions of a text as it is replayed
— and records the vector at each step, or its coordinate along a
direction. `project` reads a captured trajectory along a direction after
the fact; `compare` sets two trajectories side by side, step by step,
and finds where they diverge; `aggregate` groups trajectories and reduces
them to a mean trajectory with its spread, or to one value per group
over a window.

A trajectory point is a vector with a `step`, so a trajectory is a
collection of vectors and every vector operation reads it; a projected
trajectory is a collection of coordinates.
""",
    ),
    Family(
        "adapter",
        "Low-rank adapters: trained against a target distribution at a decision point, merged into a checkpoint, published.",
        """\
`train` fits a LoRA adapter that moves what the model says at a decision
point toward a target distribution over outcomes, and produces it as an
object whose lineage is the training's methods section. `merge` collapses
a model's adapter stack into one standalone checkpoint; `publish` puts an
adapter on the Hugging Face hub with a model card that carries its bench
provenance.

An adapter reaches a model-running node on that node's `adapter` port,
fused for that node only on top of whatever adapters the model reference
already carries.

`measure` reads an adapter as DATA — per module, what training wrote
there and how concentrated it is — which is the `weights` family's
question asked of a delta rather than of a model.
""",
    ),
    Family(
        "weights",
        "The model's own parameters, read as data: what the model IS, rather than what it did on an input.",
        """\
Every other family reads a forward pass. This one reads the learned
matrices themselves — no prompt, no sampling, nothing to be
representative of.

`capture` names parameter points in the module tree
(`layers.12.self_attn.q_proj`, `embed_tokens`, `layers.*.mlp.down_proj`)
and produces one `weights/parameter` per tensor: its shape, its norm, how
much of it is zero, and, when asked, its spectrum and its values. The
reduced forms are the default because a parameter is large — a 4B
model's embedding table is 400 million numbers, and the interesting
facts about it are a few dozen.

A question about what TRAINING changed, rather than what the model is,
belongs to `adapter/measure`: an adapter's delta is already the
difference, and reading it needs neither the model nor its base.
""",
    ),
    Family(
        "tools",
        "The tools a model may call during a conversation.",
        """\
A tool is an operation a model invokes by name mid-conversation: `calc`
evaluates arithmetic, `lookup` fetches a stored object from the bench.
Tools emit no kind of their own — what they return goes into the
transcript that called them.
""",
    ),
    Family(
        "sandbox",
        "The sandboxed filesystem a conversation's tools run in: its image, its snapshots, and each call made inside it.",
        """\
A platform family: its kinds are written by the sandbox itself, not by an
operation. An image is what a session starts from; a snapshot is a
directory as a value, entries sorted by path, so two runs that produced
the same tree produced the same object; a call is one tool invocation
with what it read and wrote.
""",
    ),
    Family(
        "provider",
        "Calls to hosted models: what was sent, what came back, what it cost, and the cassette that replays them.",
        """\
A platform family: the record of every request a conversation or chat
made to a hosted model, so a run that spent money can be replayed without
spending it again. A completion is one reply with its call's provenance;
a cassette holds recorded replies keyed by request hash.
""",
    ),
    Family(
        "model",
        "How a model is named and where a published one lives.",
        """\
A platform family. A model reference is a base and the adapters that are
part of what it means, fused before anything else; it is what a
protocol's `model` parameter names and what every space's `model` records.
A pointer says where a merged and published checkpoint lives, so a later
reference can load it as a base.
""",
    ),
    Family(
        "run",
        "What a protocol run leaves behind: every node's result, the manifest, the spend.",
        """\
A platform family: the run result is written by the executor, one per
run, naming every node's stored object. A ladder is the older experiment
record of a sequence of training rungs.
""",
    ),
)

BY_FAMILY: dict[str, Family] = {f.name: f for f in FAMILIES}

__all__ = ["BY_FAMILY", "FAMILIES"]
