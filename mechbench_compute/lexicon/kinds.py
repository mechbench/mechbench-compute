"""The kinds: every data type an op reads or produces, declared once.

A kind is singular (docs/LEXICON.md §4). Plurality is the one container,
`collection`: `{kind: "collection", item_kind, key, items, ...header}`,
where the item kind declares its `key` (the fields that identify an
item) and its `header` (what a collection of it carries). `emit` sorts a
collection's items by their key before hashing, so the same items in any
order are the same bytes.

Value types — `space`, `token`, a distribution's `top` and `tracked` —
are fields, not kinds; `values.py` declares their shapes and prose,
`shapes.py` constructs them and every op builds its items through it.
The lattice (§6) is declared with `extends`.

Names retired on 2026-09-14 (task 000496) resolve through `KIND_ALIASES`
until the release named in `lexicon.ALIASES_REMOVED_IN`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import COLLECTION, KIND_ROOT, Kind, Metric, P
from mechbench_compute.lexicon.values import (  # noqa: F401 — re-exported for the readers that import them here
    COORDS,
    F,
    ID,
    SPACE,
    SPACE_DOC,
    TOKEN,
    TOP,
    TRACKED,
    VEC,
)

DIST = F("object", "A `logits/distribution`: `{entropy_bits, top, tracked?}`.",
         properties={"entropy_bits": {"type": "number"}, "top": TOP, "tracked": TRACKED})

_TABLE_RENDERER = {"primitive": "table", "field_map": {"rows": "items"}}

# --- records --------------------------------------------------------------------------

RECORD = Kind(
    "records/record",
    "The root record: an id, the coordinates it belongs to, and whatever fields the op that made it wrote.",
    fields={"id": ID, "coords": COORDS},
    required=("id",),
    key=("id",),
    header={"segments": "When the collection was made by `records/union`: the ports it came from and how many records each contributed."},
    renderer=_TABLE_RENDERER,
    doc="The root of the lattice: a condition, a pair, a document, a vector and a grid extend it, so a port "
        "typed `records/record` takes any of them. The ops that reshape and summarise records — `select`, "
        "`rename`, `union`, `delta`, `stats`, `sum`, `top-k`, `histogram`, `table`, `chart` — take any "
        "collection at all, whatever its item kind, since every item has an id and its fields. Fields beyond "
        "`id` and `coords` are whatever the producing op wrote; a consumer that needs one under another name "
        "gets it through `records/rename`, never through a parameter.",
    metrics=(
        Metric("hamming", "distance", True,
               "How many coordinate axes two records differ on; an axis one of them lacks counts as a difference. The design's own factor structure, as a distance."),
    ),
)

CONDITION = Kind(
    "records/condition",
    "A prompt for a model, chat-shaped: a user turn, an optional system prompt, and an optional prefill that begins the assistant's turn.",
    extends="records/record",
    fields={
        "user": F("string", "The user turn.", **{"x-mechbench-text": True}),
        "system": F("string", "The system prompt, when there is one."),
        "prefill": F("string", "Text the assistant's turn begins with, so a read happens at the first token after it."),
        "tracked": F("object", "Named tokens a read reports on, by name; the first is the target.", additionalProperties={"type": "string"}),
        "template": F("boolean", "`false` to tokenize the record raw rather than through the chat template."),
    },
    required=("id", "user"),
    key=("id",),
    renderer=_TABLE_RENDERER,
    doc="Every model-running op renders a condition the same way: `system` and `user` through the model's chat "
        "template as one user turn, the assistant's turn begun with `prefill`, so the decision point — where "
        "`logits/read` reads — is the first token after the prefill, and every capture, sweep and lens can "
        "read there too. `records/fill` writes conditions from a design; a record carrying only `text` or "
        "`prompt` is tokenized raw instead, as is one that says `template: false`.",
)

PAIR = Kind(
    "records/pair",
    "Two prompts that differ in one place, for the ops that run both and compare.",
    extends="records/record",
    fields={"a": F("string", "The first prompt (the clean one, for tracing)."),
            "b": F("string", "The second prompt (the corrupted one, for tracing).")},
    required=("id", "a", "b"),
    key=("id",),
    doc="Written before the typology as `clean`/`corrupt`; those names are still read.",
)

TABLE = Kind(
    "records/table",
    "A presentation table: named, typed columns and rows of scalars. Singular; its rows are typed by its columns, not by a kind.",
    fields={
        "name": F("string", "A label for the table."),
        "description": F("string", "Free text beside the name."),
        "row_axis": F("string", "What one row stands for."),
        "columns": F("array", "`{name, dtype}` per column, in order.", items={"type": "object"}),
        "rows": F("array", "One object per row, keyed by column name.", items={"type": "object"}),
        "n_missing": F("integer", "Records skipped for lacking the summarised field, when any were."),
    },
    required=("columns", "rows"),
    renderer={"primitive": "table", "field_map": {"rows": "rows"}},
    doc="A table is for reading, not for further computation: its rows are plain objects typed by `columns`, "
        "not items of a kind, so nothing downstream reads a table but a chart and a person. `records/tabulate` "
        "makes one from any collection (coordinates become the leading columns), and `records/summarize` produces one "
        "directly.",
)

HISTOGRAM = Kind(
    "records/histogram",
    "Counts of a numeric field in fixed, equal-width bins, with the counts that fell outside.",
    fields={"bins": F("array", "The count per bin, in order.", items={"type": "integer"}),
            "below": F("integer", "Values below the first bin."),
            "above": F("integer", "Values at or above the last bin's upper edge.")},
    required=("bins", "below", "above"),
)

SUM = Kind(
    "records/sum",
    "The exact sum of a numeric field over a collection, with the count.",
    fields={"n": F("integer", "How many values were summed."), "sum": F("number", "Their sum.")},
    required=("n", "sum"),
)

CHART = Kind(
    "records/chart",
    "A chart of a table, as a stored object: mark, encoding, and either a reference to the data or the rows inline.",
    fields={"title": F("string", "The chart's title."),
            "mark": F("string", "`bar`, `line`, or `point`."),
            "encoding": F("object", "`{x, y, series?}` — which fields go on which axis."),
            "source": F("string", "The stored table the chart draws, when the executor knew it."),
            "data": F("object", "`{rows}` inline, when it did not.")},
    required=("mark", "encoding"),
)

WORD_LIST = Kind(
    "text/word-list",
    "A list of words — for the generators that sample from one, the tokenizer measures, and, with `weights`, a word-frequency table.",
    fields={"words": F("array", "The words.", items={"type": "string"}),
            "weights": F("object", "Word → count or weight, when the list is a frequency table; its keys are the words."),
            "language": F("string", "The list's language, when known."),
            "description": F("string", "Where the list came from.")},
)

# --- text ------------------------------------------------------------------------------

DOCUMENT = Kind(
    "text/document",
    "One generated or collected text, with the trace of how it was made when kept at trace fidelity. A record whose text is in `text`, so every op that reads records reads documents.",
    fields={
        "text": F("string", "The text.", **{"x-mechbench-text": True}),
        "metadata": F("object", "`sampling`, the model's wire form, tool runs — what the producing op recorded; documents made before 2026-09 keep their `coords` here too."),
        "trace": F("object", "At trace fidelity: `token_ids`, `text`, `offsets`, `generation_spans`."),
        "segmentations": F("array", "Named spans over the trace: which tokens are prompt, which are body.", items={"type": "object"}),
    },
    required=("id", "text"),
    extends="records/record",
    key=("id",),
    header={"name": "A label for the collection.", "description": "Free text beside the name.",
            "fidelity": "`text`, `segments` or `trace`: how much of each document was kept.",
            "summary": "For a remote run: calls, cost, cache hits.", "spend": "What the run bought from providers."},
    renderer={"primitive": "text", "field_map": {"text": "text"}},
    doc="What `text/generate` and `text/chat` write, one per completion. The collection's `fidelity` says how "
        "much was kept: `text` alone, `segments` (which spans are prompt and which are body), or `trace` (the "
        "token ids and offsets, which `text/score` and a positions trajectory need). A document is a record, so a "
        "corpus flows into `text/measure`, `records/select` and `activations/capture` unchanged.",
)

TRANSCRIPT = Kind(
    "text/transcript",
    "One multi-party conversation: its messages in order, each with who said it and the role each side saw it as.",
    fields={
        "id": ID,
        "messages": F("array", "`{index, participant, role_as_seen, text, call?, tool_calls?, channel?}`, in order.", items={"type": "object"}),
        "stopped": F("string", "Why the conversation ended: `max_turns`, or a stop phrase."),
        "participants": F("array", "The participant names.", items={"type": "string"}),
        "text": F("string", "The transcript rendered as text, for the browser.", **{"x-mechbench-text": True}),
    },
    required=("id", "messages"),
    key=("id",),
    header={"name": "A label for the collection.", "description": "Free text beside the name.",
            "spend": "What the run bought from providers."},
    renderer={"primitive": "chat", "field_map": {"messages": "messages"}},
    doc="What `text/converse` writes: every message in order, each naming the participant who said it and "
        "the role each side saw it as, with any tool calls it made. `stopped` records why the conversation ended "
        "— the turn cap, or a stop phrase — so a transcript that ended early says so itself.",
)

ANNOTATION = Kind(
    "text/annotation",
    "One value anchored to a span of tokens in a document of a collection.",
    fields={"anchor": F("object", "`{item_id, token_start, token_end}` — where in which document."),
            "value": F("number", "The value at that span.")},
    required=("anchor", "value"),
    key=("anchor",),
    header={"name": "A label for the layer.", "description": "Free text beside the name.",
            "collection": "The stored collection the annotations are over.",
            "value_type": "`numeric` or `categorical`.",
            "required_fidelity": "The fidelity the collection must have been kept at."},
    doc="An annotation layer sits beside a document collection rather than inside it: the collection is left "
        "as stored and the layer points into it by document id and token span. `text/score` writes one, a "
        "surprisal per token; a viewer draws the layer over the text it annotates.",
)

TOKENIZATION = Kind(
    "text/tokenization",
    "How a tokenizer splits a set of items as continuations of a prefix: depth histogram, fragmentation, script composition, and an optional gate.",
    fields={
        "tokenizer": F("string", "Which tokenizer measured."),
        "prefix": F("string", "The prefix the items were tokenized after."),
        "n_items": F("integer", "How many items."),
        "mean_depth": F("number", "Mean tokens per item after the prefix."),
        "min_depth": F("integer", "Fewest tokens any item took."),
        "max_depth": F("integer", "Most tokens any item took."),
        "single_token_fraction": F("number", "Share of items that are one token."),
        "mean_tokens_per_word": F("number", "Fragmentation against whitespace words."),
        "fragmented_fraction": F("number", "Share of items with more tokens than words."),
        "rows": F("array", "The depth histogram: `{depth, count, share}`.", items={"type": "object"}),
        "script_composition": F("object", "Unicode script → share of characters."),
        "boundary_failures": F("array", "Items that changed the prefix's own tokenization.", items={"type": "string"}),
        "gate": F("object", "`{expected_depth, pass, n_violations, violations}`, or null."),
        "most_fragmented": F("array", "The most fragmented items, with their pieces.", items={"type": "object"}),
        "items": F("array", "Every item's own measurement, when kept.", items={"type": "object"}),
    },
    required=("tokenizer", "n_items", "mean_depth", "rows"),
    renderer={"primitive": "table", "field_map": {"rows": "rows"}},
    doc="The measurement to take before a decision read: a set of outcomes the tokenizer splits into several "
        "pieces cannot be compared at one token, and a prefix whose own tokenization changes when an item "
        "follows it moves the decision point. The `gate`, when asked for, says whether every item met the "
        "expected depth, and names the ones that did not.",
)

AGENT = Kind(
    "text/agent",
    "A conversation participant: which model, what it was told, what it may call.",
    fields={"name": F("string", "The participant's name."), "model": F("string", "Its model reference."),
            "system": F("string", "Its system prompt."), "tools": F("array", "Tools it may call.", items={}),
            "budget_usd": F("number", "Its own spending cap, for a hosted model.")},
    required=("name", "model"),
    key=("name",),
    platform=True,
)

# --- logits ----------------------------------------------------------------------------

DISTRIBUTION = Kind(
    "logits/distribution",
    "A summary of a next-token distribution: its entropy, the most likely tokens, and the tokens the caller asked about.",
    fields={"entropy_bits": F("number", "The distribution's entropy in bits."),
            "top": TOP, "tracked": TRACKED},
    required=("entropy_bits", "top"),
    doc="Every op that reads a next-token distribution produces this shape or a kind that extends it: "
        "the same `top` and `tracked`, spelled once. Two reads compare over the union of the tokens "
        "they carry, the mass neither names counted as one last bucket.",
    metrics=(
        Metric("jensen-shannon", "distance", True,
               "The Jensen–Shannon distance: the square root of the divergence in bits, in [0, 1]."),
        Metric("hellinger", "distance", True, "The Hellinger distance, in [0, 1]."),
        Metric("total-variation", "distance", True,
               "Half the L1 distance between the two: the largest difference in the probability of any event."),
        Metric("kl", "distance", False,
               "KL(p ‖ q) in bits, the row against the column. Not symmetric, so a tree refuses it."),
    ),
)

DECISION = Kind(
    "logits/decision",
    "The next-token distribution at a record's decision point, with the mass on each candidate outcome and an optional best-first expansion of complete outcomes.",
    extends="logits/distribution",
    fields={"id": ID, "coords": COORDS,
            "rollout": F("object", "The expanded complete outcomes, when a rollout was asked for.")},
    required=("id", "entropy_bits", "top"),
    key=("id",),
    header={"model": "The model read.", "top_k": "How many tokens `top` holds."},
    renderer=_TABLE_RENDERER,
    doc="What `logits/read` produces, one per condition, and what `eval/expect` judges. `tracked` holds "
        "each named token by the name the protocol gave it — the op's `tracked` param, or the record's own, "
        "which takes precedence — and the first is the target. `rollout`, when asked for, expands the most "
        "likely complete outcomes past the first token.",
)

FUNNEL = Kind(
    "logits/funnel",
    "One layer of a record's commitment funnel: the next-token distribution when that layer's residual is read through the unembedding.",
    extends="logits/distribution",
    fields={"id": ID, "coords": COORDS, "layer": F("integer", "The layer read.")},
    required=("id", "layer", "entropy_bits", "top"),
    key=("id", "layer"),
    header={"name": "A label for the collection.", "description": "Free text beside the name.",
            "layers": "The layers read, in order.", "top_k": "How many tokens `top` holds."},
    renderer={"primitive": "table", "field_map": {"rows": "top"}},
    collection_renderer={"primitive": "series", "field_map": {"rows": "items", "x": "layer", "y": "entropy_bits", "label": "id"}},
    doc="Read a record's items in layer order and you see the funnel: entropy falling, one token taking over.",
)

LENS = Kind(
    "logits/lens",
    "A record's logit lens over the whole prompt: the target token's log-probability and rank at every (layer, position).",
    extends="activations/grid",
    fields={"target": TOKEN},
    required=("id", "axes", "measures", "tokens", "target"),
    key=("id",),
    header={"layers": "The layers read, in row order."},
    doc="Axes `[layer, position]`; measures `logprob` and `rank` (0 is the top readout).",
)

ATTRIBUTION = Kind(
    "logits/attribution",
    "A record's direct logit attribution: each component's contribution to the target logit, with the additivity check that says the pieces sum to the truth.",
    extends="activations/grid",
    fields={"target": TOKEN, "contrast": TOKEN,
            "per_head": F("array", "`{layer, contributions[head]}` for the layers split by head.", items={"type": "object"}),
            "additivity": F("object", "`{summed, true_logit, residual}` — the honesty number.")},
    required=("id", "axes", "measures", "target", "additivity"),
    key=("id",),
    header={"components": "The component names: `embed`, then `L0`, `L1`, …", "apply_ln": "Whether the final norm was folded in.",
            "layers": "The layers decomposed (all of them)."},
    doc="Axis `[component]`, in the header's `components` order; measure `contribution`.",
)

# --- activations -----------------------------------------------------------------------

VECTOR = Kind(
    "activations/vector",
    "One vector from a model's activation space, tagged with the space it lives in and what it was read from.",
    extends="records/record",
    fields={"space": SPACE, "vector": VEC,
            "norm": F("number", "The vector's magnitude."),
            "token": TOKEN,
            "n_pooled": F("integer", "How many positions went into it, when pooled.")},
    required=("space", "vector"),
    key=("id", "space"),
    header={"point": "The hook point read.", "source": "`resid`, `queries` or `keys`.",
            "position": "Which position, or `pooled`.", "pool": "The pooling, when pooled.",
            "layers": "The layers captured.", "d_model": "The vector width.",
            "model": "The model's wire form.",
            "skipped_empty": "Records dropped for having no text, when any.",
            "segments": "When made by `records/union`: the ports and how many each contributed."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="A grouping is a coordinate (`coords.genre`), never a `label` field; the ops that group take an `axis`. "
        "Two vectors compare only within one space; a metric refuses two that differ, naming both spaces.",
    metrics=(
        Metric("cosine", "similarity", True,
               "The cosine of the angle between the two, in [−1, 1]. With `center`, the collection's mean vector is "
               "subtracted first: transformer activations occupy a narrow cone around one dominant direction, and raw "
               "cosine measures that cone before it measures the items.",
               options=(P("center", "bool", "Subtract the collection's mean vector before comparing.", False),)),
        Metric("euclidean", "distance", True, "The straight-line distance between the two."),
        Metric("dot", "similarity", True, "The dot product, unnormalised; a tree over it uses max − dot as the distance."),
    ),
)

COORDINATE = Kind(
    "activations/coordinate",
    "A vector's scalar coordinate along a direction, with the space and the direction's identity.",
    fields={"id": ID, "coords": COORDS, "space": SPACE,
            "direction": F("object", "`{space, method, …}` — the direction projected onto, without its vector."),
            "coord": F("number", "The dot product with the unit direction."),
            "step": F("integer", "The step, when read along a trajectory."),
            "position": F("integer", "The position, when read along a trajectory."),
            "token": TOKEN},
    required=("space", "direction", "coord"),
    key=("id", "step", "space"),
    header={"axis": "For a projected trajectory: `layers` or `positions`.",
            "projected": "Always true: a coordinate collection is a projected one."},
    renderer=_TABLE_RENDERER,
    doc="What `direction/project` and `trajectory/project` emit: one number per vector, the dot product with a "
        "unit direction, with the space and the direction's derivation carried so the number can be read back "
        "to what it measures. A projected trajectory is a collection of these keyed by step; the vector itself "
        "is not kept, which is the point.",
)

GRID = Kind(
    "activations/grid",
    "A scalar field over model axes for one record: declared axes, named measures indexed in axis order, and the tokens when an axis is position.",
    extends="records/record",
    fields={"axes": F("array", "The axes, in value-index order: `layer`, `position`, `head`, `query`, `key`, `component`.", items={"type": "string"}),
            "measures": F("object", "Measure name → values, a nested list indexed in `axes` order."),
            "tokens": F("array", "The prompt's tokens, when an axis is `position`.", items={"type": "string"}),
            "error": F("string", "Why the record could not be measured, when it could not; then `measures` is empty.")},
    required=("id", "axes", "measures"),
    key=("id",),
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="The ancestor of every map an op draws over the model: a lens, an attribution, a divergence, an "
        "attention pattern, a trace, a head-ablation grid. `axes` says what the nested `measures` are indexed "
        "by, in order, so a reader (or a chart) knows that `measures.logprob[3][7]` is layer 3, position 7. A "
        "record that could not be measured carries `error` and empty measures rather than being dropped, so "
        "a grid collection has one item per input record.",
)

DIVERGENCE = Kind(
    "activations/divergence",
    "For a matched pair, 1 − cosine between the two residual streams at every (layer, position).",
    extends="activations/grid",
    required=("id", "axes", "measures"),
    key=("id",),
    header={"point": "The residual compared.", "layers": "The layers, in row order."},
    doc="Axes `[layer, position]`; measure `divergence`; `tokens` are prompt `a`'s.",
)

ATTENTION = Kind(
    "activations/attention",
    "A record's attention weights at chosen layers: per head, a matrix over the prompt's positions.",
    extends="activations/grid",
    required=("id", "axes", "measures", "tokens"),
    key=("id",),
    header={"n_heads": "Heads per layer.", "layers": "The layers captured, in axis order."},
    doc="Axes `[layer, head, query, key]`; measure `weight` — row = the attending position, column = the attended-to position.",
)

# --- geometry --------------------------------------------------------------------------

SIMILARITY = Kind(
    "geometry/similarity",
    "The pairwise matrix of a collection's items under one of their kind's metrics — cosine over vectors, Jensen–Shannon over distributions, hamming over records — one item per group, with the metric and its options recorded and, when the items carry a value on the grouping axis, how well the groups separate.",
    fields={"group": F("string", "The group's name: `layer=12`, `layer=3,head=1`, `genre=noir`, or `all`."),
            "layer": F("integer", "The layer, for a group of vectors from one space."),
            "head": F("integer", "The head, for a per-head group."),
            "space": SPACE,
            "ids": F("array", "The item ids, in matrix order.", items={}),
            "labels": F("array", "The items' values on the grouping `axis`, in matrix order.", items={}),
            "matrix": F("array", "The metric's value, `[i][j]`, at full precision.", items={"type": "array"}),
            "pairs": F("array", "Every pair `{a, b, value}`, most alike first, for a group of at most thirty-two items.", items={"type": "object"}),
            "separation": F("object", "`{intra, inter, gap}`: the mean value within groups, between groups, and how much the groups stand apart."),
            "nn_purity": F("number", "Share of items whose nearest neighbour shares their group."),
            "silhouette": F("number", "The silhouette score, when computable.")},
    key=("group",),
    header={"metric": "The metric's name.", "metric_kind": "`similarity` or `distance`: which way larger means.",
            "symmetric": "Whether m(a, b) = m(b, a).", "options": "The metric's options as applied.",
            "over": "The kind of the items compared.", "by": "The header axis the groups were formed on.",
            "axis": "The coordinate the separation reads.",
            "position": "Which position the vectors were read at, when vectors.", "point": "The hook point, when vectors."},
    doc="Written before 2026-09-15 by `direction/similarity` as `{cosine}` or `{names, cosines, norms, pairs}`; "
        "those objects stay as stored and are read by the same fields.",
)

MST = Kind(
    "geometry/mst",
    "The minimum spanning tree over a group's pairwise distances, whatever metric produced them: the spread's scale and clumpiness, the bridges between clusters, and the edges.",
    fields={"group": F("string", "The group's name, from the similarity it was built on."),
            "layer": F("integer", "The layer, for a per-layer group."), "head": F("integer", "The head, for a per-head group."),
            "n": F("integer", "Items in the group."), "n_edges": F("integer", "Edges in the tree."),
            "mean": F("number", "Mean edge: the scale of the spread."), "variance": F("number", "Edge variance: the clumpiness."),
            "stdev": F("number", "Edge standard deviation."), "cv": F("number", "stdev / mean, scale-free."),
            "total": F("number", "Sum of edge weights."), "max": F("number", "Longest edge."), "min": F("number", "Shortest edge."),
            "bridge_threshold": F("number", "Edges above this count as bridges."), "bridges": F("integer", "Edges above the threshold."),
            "components_after_cut": F("integer", "Clusters left when the bridges are cut."),
            "ids": F("array", "The item ids, in edge-index order.", items={}), "labels": F("array", "Their labels.", items={}),
            "edges": F("array", "`[i, j, weight]` per edge, in the order the tree grew.", items={"type": "array"})},
    required=("n", "n_edges"),
    key=("group",),
    header={"name": "A label for the summary.", "metric": "The metric the distances came from.",
            "options": "The metric's options as applied.", "over": "The kind of the items compared.",
            "bridge_sigma": "The bridge threshold in standard deviations.", "axis": "The coordinate the labels read."},
)

# --- intervene -------------------------------------------------------------------------

SPEC = Kind(
    "intervene/spec",
    "An intervention as an object: the spec items — where in the forward pass to act, at which layers and positions, and what to do there — declared once and wired to whichever op runs the forward pass.",
    fields={"items": F("array", "The spec items, in the grammar `intervene/apply` documents under *Spec items*.", items={"type": "object"}),
            "description": F("string", "What the intervention is for, in a sentence.")},
    required=("items",),
    doc="The same object wires to `intervene/apply` (read the next-token distribution or a capture), to "
        "`text/generate` and `text/chat` (read what the model then writes), and to any op that takes an "
        "`intervention` port — one declared intervention, several readouts. Directions and sources "
        "an item needs arrive on the consuming node's `direction` and `source` ports, which fill any item "
        "that names none; a stored spec carries no vectors of its own. Stored on the bench like a corpus, "
        "by hand or from a protocol's output.",
)

READOUT = Kind(
    "intervene/readout",
    "What one record's forward pass read out under one strength of an intervention: a next-token distribution. A capture readout is instead an `activations/vector` collection, `factor` on each vector.",
    extends="logits/distribution",
    fields={"id": ID, "coords": COORDS,
            "factor": F("number", "The sweep factor; 0 is the control. For a steer sweep, the alpha."),
            "position": F("integer", "Before 0.110.0, for a capture: the position read."),
            "captures": F("object", "Before 0.110.0, for a capture: a collection of `activations/vector`, one per hook point. A capture readout is now itself that collection.")},
    required=("id", "factor"),
    key=("id", "factor"),
    header={"spec": "The intervention items as run, with objects replaced by their provenance.",
            "sweep": "The factors run, including 0 when a control was added.",
            "readout": "`decision` or `capture`.",
            "layer": "For a steer sweep: the injection layer.",
            "direction": "For a steer sweep: the axis and values, norm and counts of the direction built."},
    renderer=_TABLE_RENDERER,
    doc="A decision readout carries `entropy_bits`, `top` and `tracked`; a capture readout carries `position` and `captures`, "
        "and a capture readout wires into another intervention's `source`.",
)

ABLATION = Kind(
    "intervene/ablation",
    "The change in a target token's log-probability when one layer's component is removed, for one record and one layer.",
    fields={"id": ID, "layer": F("integer", "The layer ablated."),
            "delta_logp": F("number", "Ablated minus baseline log-probability.")},
    required=("id", "layer", "delta_logp"),
    key=("id", "layer"),
    header={"component": "What was removed at each layer.", "layers": "The layers swept.",
            "n_conditions": "How many records.",
            "n_off_top1": "How many conditions tracked a target the model would not itself have said (each such condition carries `own_top1`). Read this before any Δ.",
            "conditions": "Per record: `{id, target, baseline_logp}` — the untouched read each delta is against.",
            "aggregates": "`{mean_delta, median_delta}` per layer across records."},
    renderer=_TABLE_RENDERER,
    doc="What `intervene/ablate-layers` produces: one item per (record, layer), the drop in the target's log-probability "
        "when that layer's sub-layer outputs are zeroed. The baseline each delta is measured against is on the "
        "header, per record, so a delta is never read without the number it is a difference from; a layer the "
        "answer runs through shows as a large negative delta.",
)

HEADS = Kind(
    "intervene/heads",
    "The mean change in a target token's log-probability with each single head zeroed: a layer × head grid over the records.",
    extends="activations/grid",
    fields={"layers": F("array", "The layers, in row order.", items={"type": "integer"}),
            "n_heads": F("integer", "Heads per layer."), "n_conditions": F("integer", "How many records."),
            "n_off_top1": F("integer", "How many conditions tracked a target the model would not itself have said."),
            "conditions": F("array", "Per record: `{id, target, baseline_logp}`.", items={"type": "object"})},
    required=("id", "axes", "measures", "layers", "n_heads"),
    doc="Axes `[layer, head]`; measure `mean_delta`. One grid for the whole record set, id `mean`.",
)

TRACE = Kind(
    "intervene/trace",
    "A pair's causal trace: how much of the clean answer's probability comes back when the clean residual is patched into the corrupted run at each (layer, position).",
    extends="activations/grid",
    fields={"target": TOKEN, "metric": F("string", "`logprob` or `prob`."),
            "value_a": F("number", "The metric on prompt `a` (clean)."),
            "value_b": F("number", "The metric on prompt `b` (corrupted), the baseline.")},
    required=("id", "axes", "measures"),
    key=("id",),
    header={"point": "The residual patched.", "metric": "`logprob` or `prob`.", "layers": "The layers, in row order."},
    doc="Axes `[layer, position]`; measure `recovery` — the change in the metric from the `b` baseline; `tokens` are prompt `b`'s.",
)

# --- direction -------------------------------------------------------------------------

DIRECTION = Kind(
    "direction/vector",
    "A unit vector in a model's activation space, with its derivation: how it was made, from what, on which model.",
    extends="activations/vector",
    fields={"unit": F("boolean", "Whether the vector is unit length."),
            "derivation": F("object", "`{method, sources, model, axis?, positive?, negative?, …}` — how it was made.")},
    required=("space", "vector", "derivation"),
    key=("id", "space"),
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="`norm` is the magnitude before normalisation, which some readings use.",
)

VOCAB = Kind(
    "direction/vocab",
    "What a direction says in token space: the distribution the unembedding gives it and its negative.",
    fields={"space": SPACE, "top_k": F("integer", "How many tokens per sign."),
            "positive": DIST, "negative": DIST},
    required=("space", "positive", "negative"),
    doc="What `direction/unembed` produces: the direction pushed through the unembedding as if it were a final "
        "residual, and its negative likewise, each read as a next-token distribution. The tokens the positive "
        "side promotes are what the axis 'says'; the negative side is what it says when reversed. Two "
        "distributions, so the distribution metrics compare a direction's vocabulary with another's.",
)

# --- trajectory ------------------------------------------------------------------------

POINT = Kind(
    "trajectory/point",
    "One step of a trajectory: the residual vector at one layer and position of one record.",
    extends="activations/vector",
    fields={"step": F("integer", "The step along the axis."),
            "position": F("integer", "The position read."),
            "vocab": DIST,
            "steps": F("array", "The window pooled, when reduced.", items={"type": "integer"})},
    required=("id", "step", "space", "vector"),
    key=("id", "step"),
    header={"axis": "`layers` or `positions`.", "point": "The residual read.", "layers": "The layers.",
            "position": "For a layers axis: which position.", "positions": "For a positions axis: which positions.",
            "d_model": "The vector width.",
            "replay": "`trace`, `text` or `mixed`.", "n_items": "How many records.",
            "max_steps": "The per-record step cap, when set.", "reduce": "The reduction, when reduced.", "steps": "The window, when reduced."},
    renderer=_TABLE_RENDERER,
    doc="A projected trajectory is a collection of `activations/coordinate` with `step` and `position`, not a point without its vector.",
)

COMPARISON = Kind(
    "trajectory/comparison",
    "Two trajectories compared at one step: cosine, angle, and the norms.",
    fields={"id": ID, "step": F("integer", "The step."), "layer": F("integer", "The layer."), "position": F("integer", "The position."),
            "cosine": F("number", "Cosine between the two vectors."), "angle_deg": F("number", "The angle in degrees."),
            "norm_a": F("number", "The first vector's norm."), "norm_b": F("number", "The second's."), "norm_ratio": F("number", "b / a.")},
    required=("step", "cosine"),
    key=("id", "step"),
    header={"axis": "The shared axis.", "pair_by": "`id` or `step`.", "threshold": "The cosine below which they count as diverged.",
            "n_pairs": "Rows paired.", "divergence_step": "The first step below the threshold.", "min_cosine_step": "The step of least agreement.",
            "per_step": "`{step, mean_cosine, n}` across ids."},
    renderer=_TABLE_RENDERER,
    doc="What `trajectory/compare` produces: the two trajectories paired by id or by step, and at each step the "
        "cosine and angle between their vectors and the ratio of their norms. The header carries the first "
        "step at which the cosine fell below the threshold — where two models, or two prompts, stop agreeing.",
)

SUMMARY = Kind(
    "trajectory/summary",
    "A group of trajectory rows reduced at one step, or over a window: the mean vector with its spread, or the mean and std of coordinates.",
    fields={"group": F("string", "The group."), "step": F("integer", "The step, for a per-step summary."), "n": F("integer", "Rows in the group."),
            "mean": F("number", "Mean coordinate."), "std": F("number", "Its standard deviation."),
            "norm": F("number", "The mean vector's norm."), "mean_norm": F("number", "Mean norm of the members."),
            "spread": F("number", "Mean cosine of members to the mean."), "vector": VEC},
    required=("group", "n"),
    key=("group", "step"),
    header={"aggregated": "`{by, as, steps}` — how the rows were grouped and reduced."},
    renderer=_TABLE_RENDERER,
    doc="What `trajectory/aggregate` produces, in one of three shapes the header's `as` names: per step, the mean "
        "coordinate and its standard deviation across the group; over a window, one value per group; or as "
        "vectors, the group's mean vector with the spread of its members around it — the shape "
        "`direction/fit` reads directly.",
)

# --- adapter ---------------------------------------------------------------------------

LORA = Kind(
    "adapter/lora",
    "A trained LoRA adapter: its weights, the base and stack it was trained on, its shape, and the training's methods section.",
    fields={"format": F("string", "`safetensors`."), "base_model": F("string", "The base it was trained on."),
            "trained_on": F("object", "`{base, adapters}` — the full stack."),
            "lora": F("object", "`{rank, alpha, scale, target_modules, params}`."),
            "train": F("object", "Steps, lr, seed, batch, final loss, the target spec, and the counts."),
            "data": F("string", "The safetensors bytes.", contentEncoding="binary")},
    required=("format", "lora", "data"),
    doc="What `adapter/train` produces, and what a model-running node takes on its `adapter` port. `train` records "
        "the whole training — steps, learning rate, seed, batch, final loss, the target it was trained toward "
        "and the counts — so the adapter's own object is the methods section of the experiment that made it, "
        "and `trained_on` names the base and the adapters it was stacked on, which is what a later fusion must "
        "match.",
)

CHECKPOINT = Kind(
    "adapter/checkpoint",
    "A merged checkpoint published to the bench or the hub: its files with their hashes, and the stack that was merged.",
    fields={"files": F("array", "`{name, size, sha256}` per file.", items={"type": "object"}),
            "merged_from": F("object", "The model reference merged, in wire form."),
            "base_snapshot": F("string", "The base revision merged onto."),
            "location": F("string", "Where it landed.")},
    required=("files", "merged_from"),
)

PARAMETER = Kind(
    "weights/parameter",
    "One of the model's learned tensors, read as data: where it sits in the module tree, its shape, and what its numbers are like.",
    extends="records/record",
    fields={"id": ID, "coords": COORDS,
            "shape": F("array", "The tensor's shape.", items={"type": "integer"}),
            "n": F("integer", "How many numbers it holds."),
            "dtype": F("string", "The dtype it is stored in (`bfloat16`, `float32`)."),
            "frobenius": F("number", "‖W‖_F — the tensor's magnitude."),
            "mean": F("number", "Mean of its values."),
            "std": F("number", "Standard deviation of its values."),
            "max_abs": F("number", "Largest magnitude in it, where an outlier channel shows."),
            "sparsity": F("number", "Share of its values that are exactly zero."),
            "singular_values": F("array", "The top singular values, largest first — only when `spectrum` asked for them.", items={"type": "number"}),
            "spectral": F("number", "σ₁, when the spectrum was computed."),
            "effective_rank": F("number", "exp(H(p)) over the spectrum, when it was computed: how many directions the tensor really uses."),
            "values": F("array", "The tensor itself, flattened — only when `values` asked for it, and only under the ceiling.", items={"type": "number"})},
    required=("id", "shape", "n", "frobenius"),
    key=("id",),
    header={"model": "The model's wire form — which weights these are.",
            "captured": "`{parameters, values, of}` — how many tensors this node read, how many numbers that is, and how many the model has."},
    renderer=_TABLE_RENDERER,
    doc="What `weights/capture` produces: one item per parameter tensor, `id` and `coords.module` naming it in the model's "
        "own tree (`layers.12.self_attn.q_proj.weight`). `coords` carry `layer`, `container`, `projection` and "
        "`parameter` where the name has them, so a reading groups by layer or by projection the way an activation "
        "capture groups by layer. The reduced forms are the default: the spectrum costs an SVD per tensor and the "
        "values are the tensor itself, so each is asked for.",
)

DELTA = Kind(
    "adapter/delta",
    "What training wrote at one module: the norm, spectrum and effective rank of an adapter's delta there, and its share of the adapter's mass.",
    extends="records/record",
    fields={"id": ID, "coords": COORDS,
            "frobenius": F("number", "‖ΔW‖_F at this module — how much was written here at all."),
            "spectral": F("number", "The largest singular value: how much of the write is one direction."),
            "singular_values": F("array", "The spectrum, largest first (`top_k` of them).", items={"type": "number"}),
            "effective_rank": F("number", "exp(H(p)) over the normalised spectrum: 1 for a single direction, r for r equal ones."),
            "mass_share": F("number", "This module's ‖ΔW‖² over the total of what was measured; the shares sum to 1."),
            "rank": F("integer", "The adapter's rank at this module — the number of directions it could have written."),
            "shape": F("array", "`[out, in]` of the module this delta is over.", items={"type": "integer"}),
            "vector": F("array", "The principal left-singular direction, unit length, in the module's output space — present when the node asked for it.", items={"type": "number"}),
            "basis": F("object", "`{module, side, d}` — which space `vector` lives in.")},
    required=("id", "frobenius", "effective_rank", "mass_share"),
    # Two adapters measured into one collection carry the same module
    # ids, so the coordinates — which name the adapter — are part of
    # the identity, as a vector's space is.
    key=("id", "coords"),
    header={"base_model": "The base the adapter was trained on.",
            "trained_on": "`{base, adapters}` — the full stack.",
            "lora": "`{rank, alpha, scale, target_modules}` of the adapter read.",
            "measured": "`{modules, layers, frobenius}` — what this node covered, which is what the shares are shares of.",
            "source": "What to call this adapter, stamped on every item's `coords.adapter`."},
    renderer=_TABLE_RENDERER,
    doc="What `adapter/measure` produces: one item per (layer, module), read from the adapter's own low-rank factors "
        "with no model and no forward pass. `coords.layer` and `coords.module` are what a reading groups on — "
        "mass by layer is the 'where did training write' map — and `coords.adapter` separates two adapters "
        "measured into one collection. With `vectors` on, two adapters' writes at the same module are compared "
        "by `geometry/compare` through their principal directions — whether two training runs moved the model "
        "the same way, answered in weight space rather than by capturing what they do to a prompt.",
    metrics=(
        Metric("cosine", "similarity", True,
               "The cosine between the two deltas' principal directions, in [−1, 1]. Needs `vectors` on the "
               "measuring node, and compares only within one module — group by `coords.module` (`by: \"module\"`), "
               "since two modules' output spaces are different spaces."),
    ),
)

PUSH = Kind(
    "adapter/push",
    "The record of an adapter published to the Hugging Face hub: the repository, the files, and the commit that can fetch it back.",
    fields={"repo": F("string", "`<namespace>/<name>`."), "private": F("boolean", "Whether the repository is private."),
            "dry_run": F("boolean", "Whether nothing was uploaded."), "files": F("array", "`{name, bytes}` per file.", items={"type": "object"}),
            "lora": F("object", "`{rank, alpha, target_modules}`."), "base_model": F("string", "The base named in the model card."),
            "commit": F("string", "The commit created."), "url": F("string", "The repository at that commit."),
            "hf_adapter_ref": F("object", "`{repo, revision}` to fetch it back.")},
    required=("repo", "files"),
)

# --- eval ------------------------------------------------------------------------------

VERDICT = Kind(
    "eval/verdict",
    "A verdict on one record: a judge's score, label or preference with every vote; or an expectation's pass with the number it was judged on.",
    fields={"id": ID, "coords": COORDS,
            "score": F("number", "Mean score, for a numeric scale."), "spread": F("number", "Standard deviation of the scores."),
            "min": F("number", "Lowest score."), "max": F("number", "Highest score."),
            "label": F("string", "The majority label, for a categorical scale."), "winner": F("string", "`A` or `B`, for a pairwise scale."),
            "counts": F("object", "Label or winner → votes."), "agreement": F("number", "Share of votes for the majority."),
            "rationale": F("string", "The first parsed vote's rationale."),
            "n_votes": F("integer", "Votes cast."), "n_parsed": F("integer", "Votes that could be read."),
            "votes": F("array", "Every vote as `{vote, parsed, order, …}`.", items={"type": "object"}),
            "unparsed": F("boolean", "True when no vote could be read."),
            "expect": F("string", "For an expectation: its type."),
            "entropy_bits": F("number", "For an expectation: the read's entropy."),
            "kl_bits": F("number", "For an expectation: KL from the expected distribution."),
            "mass": F("number", "For an expectation: the probability mass on the named outcomes."),
            "p_expected": F("number", "For an `answer` expectation: the expected token's probability."),
            "pass": F("boolean", "Whether the expectation was met; null when it could not be judged."),
            "note": F("string", "Why it could not be judged, when it could not.")},
    required=("id",),
    key=("id",),
    header={"judge": "Who graded, on what scale, with what rubric.",
            "summary": "For a judge: mean/median/stdev or counts, `n_unparsed`, the position-bias diagnostic. For an expectation: `pass_rate`, `n_pass`, `n_judged`, `n_unjudgeable`.",
            "spend": "What the judging cost.", "name": "A label for the collection.", "description": "Free text beside the name."},
    renderer=_TABLE_RENDERER,
    doc="One kind for every evaluation, so a verdict from a judge and one from an expectation sit in one table "
        "and one rate. A judge's verdict keeps every vote with the order the options were shown in, so a "
        "position bias can be seen rather than suspected; an expectation's keeps the number it was judged on "
        "(the entropy, the KL, the mass) beside the pass, and says why when it could not be judged at all.",
)

# --- platform kinds (not produced by ops) -------------------------------------------------

PLATFORM: tuple[Kind, ...] = (
    Kind("sandbox/image", "A sandbox image: base, tools, limits, and the tree it starts from.", platform=True,
         doc="What a sandboxed conversation starts from. Two sessions on the same image start from the same tree, "
             "so their snapshots differ only by what the tools did."),
    Kind("sandbox/snapshot", "A directory as a value: entries sorted by path, mounts by identity.", platform=True,
         renderer={"primitive": "table", "field_map": {"rows": "entries"}},
         doc="A filesystem tree as a stored object, so what a session's tools wrote is content-addressed like every "
             "other result: the same tree from two runs is the same object."),
    Kind("sandbox/call", "One tool call inside a sandbox session, with what it read and wrote.", platform=True,
         doc="The record of one invocation: which tool, with what arguments, and the files it read and wrote."),
    Kind("provider/cassette", "Recorded provider responses keyed by request hash, for replay.", platform=True,
         doc="The replies a run received from hosted models, keyed by the hash of the request that got them, so a "
             "re-run replays the same replies without calling the provider again."),
    Kind("provider/call", "The provenance of one provider call: model version, usage, cost, latency.", platform=True,
         doc="What one request to a hosted model cost and what served it — the version the provider reported, the "
             "tokens in and out, the time taken."),
    Kind("provider/completion", "One provider reply: text, parts, stop reason, usage, and its call.", platform=True,
         doc="One reply as the provider returned it, with the call that produced it, before it became a document "
             "or a transcript message."),
    Kind("model/ref", "A model reference: a base and the adapters that are part of what it means.", platform=True,
         doc="What a protocol's `model` parameter names and what every space's `model` records. The adapters a "
             "reference carries are fused before anything else; an adapter arriving on a node's port is fused on "
             "top, for that node only."),
    Kind("model/pointer", "Where a published model lives, so a later reference can load it.", platform=True,
         doc="What `adapter/merge` leaves behind: the location of a merged checkpoint, on the bench or on the hub, "
             "usable as the base of a later model reference."),
    Kind("run/ladder", "A ladder of rungs, as the older experiments recorded one.", platform=True,
         doc="A sequence of training rungs recorded as one object, from before protocols were the unit of a "
             "run; kept so those results still read."),
    Kind("run/result", "A protocol run's result: every node's path, the manifest, the spend.", platform=True,
         doc="Written by the executor when a run completes: the stored object of every node, keyed by node id, "
             "with the manifest that fingerprints the run and what it spent."),
    AGENT,
)

COLLECTION_KIND = Kind(
    COLLECTION,
    "The one container: items of one kind, identified by the kind's key, sorted by that key when stored, with the header fields the kind declares.",
    fields={"item_kind": F("string", "The kind of every item."),
            "key": F("array", "The item fields that identify an item.", items={"type": "string"}),
            "items": F("array", "The items.", items={"type": "object"})},
    required=("item_kind", "key", "items"),
    doc="Every plural result is this one shape. The item kind declares the `key` — the fields that identify an "
        "item — and its header, the collection-level facts that ride with the items (a model, a metric, a "
        "pass rate). Items are sorted by key before the object is hashed, so the same items in any order are "
        "the same object; order that matters is a property of the key (`step`, `layer`), never of the "
        "container. A port declared as `collection` takes any collection at all.",
)

KINDS: tuple[Kind, ...] = (
    RECORD, CONDITION, PAIR, TABLE, HISTOGRAM, SUM, CHART, WORD_LIST,
    DOCUMENT, TRANSCRIPT, ANNOTATION, TOKENIZATION,
    DISTRIBUTION, DECISION, FUNNEL, LENS, ATTRIBUTION,
    VECTOR, COORDINATE, GRID, DIVERGENCE, ATTENTION,
    SIMILARITY, MST,
    SPEC, READOUT, ABLATION, HEADS, TRACE,
    DIRECTION, VOCAB,
    POINT, COMPARISON, SUMMARY,
    LORA, CHECKPOINT, DELTA, PUSH,
    PARAMETER,
    VERDICT,
    *PLATFORM,
    COLLECTION_KIND,
)

BY_KIND: dict[str, Kind] = {k.name: k for k in KINDS}

#: Retired kind names (2026-09-14, task 000496): old string -> (new kind
#: name, whether the old object was a collection of it). A plural old
#: kind maps to its item kind and `True`.
KIND_ALIASES: dict[str, tuple[str, bool]] = {
    # A `document_collection` was a collection of documents; resolving it
    # to the bare container told a `records/record` port it had been
    # wired "a collection of `collection`" and refused every corpus
    # generated before the typology.
    "document_collection": ("text/document", True),
    "record_set": ("records/record", True),
    # What the experiment authors wrote on their prompt objects.
    "records": ("records/record", True),
    "~canonical/kinds/text": ("text/document", False),
    "~canonical/kinds/transcript": ("text/transcript", False),
    "transcript": ("text/transcript", False),
    "~canonical/kinds/conversation": ("text/transcript", False),
    "~canonical/kinds/lens-trajectory": ("logits/funnel", False),
    "~canonical/kinds/lens-trajectory/2": ("logits/funnel", False),
    "~canonical/kinds/condition-set": ("records/condition", True),
    "~canonical/kinds/agent": ("text/agent", False),
    "agent": ("text/agent", False),
    "~canonical/kinds/trajectory": ("trajectory/point", True),
    "~canonical/kinds/tokenizer-stats": ("text/tokenization", False),
    "~canonical/kinds/annotated-tokens": ("text/annotation", True),
    "~canonical/kinds/embeddings": ("activations/vector", True),
    "embeddings": ("activations/vector", True),
    "~canonical/kinds/call-provenance": ("provider/call", False),
    "~canonical/kinds/fs-snapshot": ("sandbox/snapshot", False),
    "fs_snapshot": ("sandbox/snapshot", False),
    "~canonical/kinds/sandbox-image": ("sandbox/image", False),
    "~canonical/kinds/sandbox-tool-call": ("sandbox/call", False),
    "~canonical/kinds/provider-cassette": ("provider/cassette", False),
    "provider_cassette": ("provider/cassette", False),
    "completion": ("provider/completion", False),
    "metric_table": ("records/table", False),
    "histogram": ("records/histogram", False),
    "viz_spec": ("records/chart", False),
    "chart_spec": ("records/chart", False),
    "word_list": ("text/word-list", False),
    "decision_read": ("logits/decision", True),
    "decision_distribution": ("logits/decision", True),
    "intervene_readout": ("intervene/readout", True),
    "ablation_sweep": ("intervene/ablation", True),
    "head_ablation": ("intervene/heads", False),
    "steer_sweep": ("intervene/readout", True),
    "patch_trace": ("intervene/trace", True),
    "attention_patterns": ("activations/attention", True),
    "logit_attribution": ("logits/attribution", True),
    "lens_map": ("logits/lens", True),
    "divergence_map": ("activations/divergence", True),
    "residual_vectors": ("activations/vector", True),
    "similarity_matrix": ("geometry/similarity", True),
    "mst_summary": ("geometry/mst", True),
    "direction": ("direction/vector", False),
    "direction_similarity": ("geometry/similarity", False),
    "direction_similarity_matrix": ("geometry/similarity", False),
    "projections": ("activations/coordinate", True),
    "direction_vocab": ("direction/vocab", False),
    "trajectory": ("trajectory/point", True),
    "trajectory_projection": ("trajectory/point", True),
    "trajectory_comparison": ("trajectory/comparison", True),
    "trajectory_summary": ("trajectory/summary", True),
    "tokenizer_stats": ("text/tokenization", False),
    "annotation_layer": ("text/annotation", True),
    "adapter": ("adapter/lora", False),
    "checkpoint_manifest": ("adapter/checkpoint", False),
    "model_pointer": ("model/pointer", False),
    "hf_push": ("adapter/push", False),
    "ladder": ("run/ladder", False),
    "pipeline_result": ("run/result", False),
}

#: Where the legacy plural objects kept their items, by old kind. The
#: old shapes are still on the bench; `items_of` reads them.
_LEGACY_ITEMS_FIELD: dict[str, str] = {
    "document_collection": "items", "record_set": "records", "records": "records",
    "decision_read": "conditions", "decision_distribution": "conditions",
    "intervene_readout": "rows", "ablation_sweep": "rows", "steer_sweep": "rows",
    "patch_trace": "pairs", "attention_patterns": "rows", "logit_attribution": "rows",
    "lens_map": "rows", "divergence_map": "pairs", "residual_vectors": "rows",
    "similarity_matrix": "layers", "mst_summary": "layers", "projections": "rows",
    "trajectory": "rows", "trajectory_projection": "rows", "trajectory_comparison": "rows",
    "trajectory_summary": "rows", "annotation_layer": "values",
}


def all_fields(kind: Kind) -> dict[str, dict[str, Any]]:
    """The kind's fields including every ancestor's, the nearest
    declaration winning."""
    chain: list[Kind] = []
    cur: Kind | None = kind
    while cur is not None:
        chain.append(cur)
        cur = BY_KIND[cur.extends] if cur.extends else None
    out: dict[str, dict[str, Any]] = {}
    for k in reversed(chain):
        out.update(k.fields)
    return out


def ancestry(name: str) -> tuple[str, ...]:
    """The kind and every kind it extends, nearest first."""
    out: list[str] = []
    cur: str | None = name
    while cur is not None and cur in BY_KIND and cur not in out:
        out.append(cur)
        cur = BY_KIND[cur].extends
    return tuple(out)


def satisfies(actual: str, declared: str) -> bool:
    """Whether an object of kind `actual` may fill a port declared as
    `declared`: the same kind, or one that extends it. `collection`
    declared means any kind at all (the port takes whatever items
    arrive, as `records/union` does)."""
    if declared == COLLECTION:
        return True
    return declared in ancestry(actual)


def canonical_kind_path(name: str) -> str:
    """The registered identity of a bare kind name."""
    if name == COLLECTION or name.startswith("~"):
        return name
    return f"{KIND_ROOT}{name}"


class RetiredKindName(DeprecationWarning):
    """A kind string that the typology renamed. The bench keeps every
    object as it was stored, so the alias table is not scheduled for
    removal; the warning is for a protocol that still writes the old
    spelling in a param or a fixture."""


_warned: set[str] = set()


def resolve_kind(kind: str, *, warn: bool = True) -> tuple[str, bool]:
    """(bare kind name, was-a-collection) for any spelling a stored
    object may carry: bare, registered, or a retired string. Raises
    `KeyError` for a string that is none of these. A retired string
    resolves and warns once per process."""
    s = kind.strip()
    if s.startswith(KIND_ROOT):
        s = s[len(KIND_ROOT):]
    if s in BY_KIND:
        return s, s == COLLECTION
    hit = KIND_ALIASES.get(kind) or KIND_ALIASES.get(s)
    if hit is None:
        raise KeyError(kind)
    if warn and kind not in _warned:
        _warned.add(kind)
        warnings.warn(
            f"kind {kind!r} is a retired spelling of {hit[0]!r}"
            + (" (a collection of it)" if hit[1] else "")
            + "; new objects carry the current name.",
            RetiredKindName, stacklevel=2)
    return hit


def item_kind_of(obj: Mapping[str, Any]) -> str | None:
    """The kind of the items in `obj`, whether it is a `collection` or a
    legacy plural object; None for a singular object."""
    k = obj.get("kind")
    if k == COLLECTION:
        ik = obj.get("item_kind")
        try:
            return resolve_kind(str(ik))[0] if ik else None
        except KeyError:
            return str(ik)
    if isinstance(k, str):
        try:
            name, plural = resolve_kind(k)
        except KeyError:
            return None
        return name if plural else None
    return None


def items_of(obj: Any) -> list[Any]:
    """The items of a collection, however it is spelled: the `collection`
    container, a legacy plural object, or a bare list."""
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, Mapping):
        raise ValueError("not a collection: a list or a mapping was expected")
    k = obj.get("kind")
    if k == COLLECTION and isinstance(obj.get("items"), list):
        return list(obj["items"])
    if isinstance(k, str) and k in _LEGACY_ITEMS_FIELD:
        return list(obj.get(_LEGACY_ITEMS_FIELD[k]) or [])
    # An unkinded object, or a container written with the list under an
    # older field name: the older names win, as they always did.
    for f in ("records", "conditions", "rows", "items"):
        if isinstance(obj.get(f), list):
            return list(obj[f])
    raise ValueError(f"not a collection: kind {k!r} carries no items")


def collection(item_kind: str, items: list[Any], **header: Any) -> dict[str, Any]:
    """Build a `collection` of `item_kind`, keyed as the kind declares."""
    kind = BY_KIND[item_kind]
    out: dict[str, Any] = {"kind": COLLECTION, "item_kind": item_kind, "key": list(kind.key), "items": list(items)}
    out.update({k: v for k, v in header.items() if v is not None})
    return out


_SPACE_ORDER = ("model", "layer", "point", "head", "d")


def _sort_value(v: Any) -> tuple[int, Any]:
    """A total order over JSON values for a key field: None first, then
    numbers, then strings, then everything else by its JSON text."""
    if v is None:
        return (0, "")
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, (int, float)):
        return (1, v)
    if isinstance(v, str):
        return (2, v)
    if isinstance(v, Mapping):
        # A space sorts by model, layer, point, head, width — the order a
        # reader thinks in — not by its field names alphabetically (which
        # is how a stored map comes back).
        keys = (_SPACE_ORDER if set(v) == set(_SPACE_ORDER) else tuple(sorted(v)))
        return (3, tuple((k, _sort_value(v[k])) for k in keys))
    if isinstance(v, (list, tuple)):
        return (4, tuple(_sort_value(x) for x in v))
    return (5, json.dumps(v, sort_keys=True, default=str))


def canonical_collection(obj: Any) -> Any:
    """`obj` with a `collection`'s items sorted by their key — the form
    that is hashed and stored, so the same items in any order are the
    same bytes. Anything that is not a collection is returned as is.
    Idempotent."""
    if not isinstance(obj, Mapping) or obj.get("kind") != COLLECTION:
        return obj
    key = list(obj.get("key") or [])
    items = list(obj.get("items") or [])
    if key:
        items = sorted(items, key=lambda it: tuple(_sort_value(it.get(k) if isinstance(it, Mapping) else None)
                                                   for k in key))
    out = dict(obj)
    out["items"] = items
    return out


__all__ = [
    "BY_KIND", "COLLECTION", "COLLECTION_KIND", "KINDS", "KIND_ALIASES", "PLATFORM",
    "all_fields", "ancestry", "canonical_collection", "canonical_kind_path", "collection",
    "item_kind_of", "items_of", "resolve_kind", "satisfies",
]
