"""Ops that run a model's forward pass: reading, editing or steering its
activations, and generating from it.

Shared conventions, stated once here and referred to from the entries:

* A record's **prompt** is its `user`, `prompt` or `text` field (the
  first present). The chat-shaped blocks (`text/generate`, `logits/decision`,
  `logits/funnel`) instead read the fields named by `system_field`,
  `user_field` and `prefill_field`.
* `template` is how a prompt is tokenized: `"raw"` as plain text,
  `"chat"` wrapped in the model's chat template as one user turn.
* `layers` is a list of layer indices or `"all"`.
* A **target** is a token whose probability the block follows. It is
  tokenized as a continuation, so include the leading space where the
  model would ("` Paris`", not "`Paris`"); a record's own `target`
  field takes precedence over the param.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Emits, Op, P

_TEMPLATE = P("template", "string",
              "How each prompt is tokenized: `\"raw\"` as plain text, `\"chat\"` "
              "wrapped in the model's chat template as a user turn.",
              "raw")

_LAYERS_ALL = P("layers", "list[int] | \"all\"",
                "Which layers to run over.",
                "all")


def _target(what: str) -> P:
    return P("target", "string",
             f"The token whose probability is followed — {what}. Tokenized as a "
             "continuation (include the leading space). A record's own `target` "
             "field takes precedence; when neither is given, the model's own "
             "top-1 prediction for that prompt is used.",
             None)


_PROMPT_INPUT = (
    "`records` — the prompts, one per record; a record's prompt is its `user`, "
    "`prompt` or `text` field. A record may carry its own `target`."
)

_FIELD_PARAMS = (
    P("system_field", "string",
      "The record field holding the system prompt. Naming the field, rather "
      "than renaming the record, is how a node consumes a `template` node's "
      "output as it is.",
      "system"),
    P("user_field", "string",
      "The record field holding the user turn. Required to be present on "
      "every record.",
      "user"),
)

_PREFILL_FIELD = P("prefill_field", "string",
                   "The record field holding text the assistant turn is begun "
                   "with, so the read happens at the first token after it — "
                   "`'{ \"name\": \"'` reads the model's first token inside a JSON "
                   "value.",
                   "prefill")


# --- editing the forward pass ------------------------------------------------------

INTERVENE = Op(
    name="intervene/apply",
    summary=(
        "Edit a model's activations at chosen points during the forward "
        "pass — zero them, patch them from another run, add or remove a "
        "direction, scale them — and read out what changed in the next-token "
        "distribution or in the activations themselves."
    ),
    description="""\
An intervention is a list of **spec items**. Each item names a **point** in
the forward pass (a residual stream, an attention or MLP output, a head's
query vector, …), which **layers** and token **positions** it applies to, and
an **op** to perform there. All items are applied together in one forward
pass over each record, and the result is read out either as a **decision** —
the next-token distribution at the last position — or as a **capture** — the
raw activations at named points.

A `sweep` runs the whole spec at several strengths, and by default a
strength‑0 **control** is added so every record has a baseline row to compare
against. The output has one row per record per sweep factor.

### Spec items

```json
{
  "point": "resid_post",
  "layers": [14],
  "positions": "last",
  "op": "add",
  "strength": 4.0,
  "direction": {"$fetch": "$direction"}
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `point` | string | `"resid_post"` | Where in the forward pass to act. Residual points: `resid_pre`, `resid_post`, `attn_out`, `mlp_out`. Inside a block: `attn.in_norm`, `attn.q`, `attn.k`, `attn.v`, `attn.q_pre_rope`, `attn.k_pre_rope`, `attn.q_pre_norm`, `attn.k_pre_norm`, `attn.scores`, `attn.weights`, `attn.per_head_out`, `attn.o_in`, `mlp.in_norm`, `mlp.gate`, `mlp.up`, `mlp.act`, `mlp.down_in`, `gate_out`. Whole-model points, which take no `layers`: `embed`, `final_norm`, `logits`. |
| `layers` | list[int] \\| `"all"` | `"all"` | Which layers the item applies to. |
| `positions` | `"last"` \\| `"all"` \\| list[int] \\| object | `"last"` | Which token positions. A list of indices (negative counts from the end); `{"tokens": ["lighthouse"]}` selects every position whose token matches; `{"range": [2, 6]}` selects a half-open span. |
| `heads` | list[int] | all | Restrict the item to these attention heads, at a point that has a head axis (`attn.q`, `attn.k`, `attn.v`, `attn.scores`, `attn.weights`, `attn.per_head_out`, and the pre-norm/pre-rope variants). |
| `neurons` | list[int] | all | Restrict the item to these indices along the feature axis — MLP neurons at `mlp.act`, residual dimensions at `resid_post`, vocabulary entries at `logits`. |
| `op` | string | `"zero"` | What to do there — see the table below. |
| `strength` | float | `1.0` | The item's magnitude: the coefficient for `add`, the factor for `scale`, the value for `clamp`, the angle in radians for `rotate`. Multiplied by each sweep factor. |
| `direction` | direction | — | The direction for `add`, `project_out`, `clamp`, `rotate` and (optionally) `patch`. May instead arrive by edge on the node's `direction` port, which fills every item that names none. |
| `direction2` | direction | — | The second axis of the plane for `rotate`. |
| `source` | record | — | A `residual_vectors` record supplying replacement activations for `mean`, `resample` and `patch`; rows are matched to the item's layer. May instead arrive by edge on the `source` (or `vectors`) port. |
| `row` | object | — | For `patch`: which row of `source` to write in, e.g. `{"index": 0}`. |
| `condition` | object | — | Apply the item only at positions whose activation projects onto a direction above (or below) a threshold: `{"direction": …, "threshold": 0.0, "above": true}`. |
| `seed` | int | the block's `seed` | The seed `resample` draws with. |

The ops:

| `op` | Effect at the selected positions | Needs |
|---|---|---|
| `zero` | Set the activation to zero. | — |
| `mean` | Replace it with the mean of the `source` rows. | `source` |
| `resample` | Replace it with a row drawn at random from `source`. | `source` |
| `patch` | Replace it with one specific `source` row, or with `direction` itself. | `source` + `row`, or `direction` |
| `add` | Add `strength × direction`. | `direction` |
| `scale` | Multiply it by `strength`. | — |
| `clamp` | Set its component along `direction` to `strength`. | `direction` |
| `project_out` | Remove its component along `direction`. | `direction` |
| `rotate` | Rotate it by `strength` radians in the plane of `direction` and `direction2`. | `direction`, `direction2` |

The older `intervene/layers` and `intervene/heads` and `intervene/steer` operations are special cases of this
grammar.
""",
    inputs="""\
`records` — the prompts to run, one forward pass each. A record's prompt is
its `user`, `prompt` or `text` field. A record may also carry its own `track`
and `outcomes`, which take precedence over the params of the same name.

`direction` (optional, by edge) — a direction record that fills any spec
item without one. `source` or `vectors` (optional, by edge) — a
`residual_vectors` record that fills any `mean`/`resample`/`patch` item
without one.
""",
    emits=Emits('intervene/readout', collection=True, doc='One item per record per factor: `id`, `coords`, `factor`, and the readout — for a decision, `entropy_bits`, `top` (the most likely next tokens with their log-probs), `track_logp` and `outcome_mass` when asked for; for a capture, `position` and `captures` (point → vector, at most 4096 values each). The header carries `spec` (the list as run, with directions and sources replaced by their provenance) and `sweep` (the factors, including `0.0` when a control was added).'),
    params=(
        P("spec", "list[object]",
          "The intervention items, applied together in one forward pass per "
          "record. At least one is required; the fields are described under "
          "*Spec items* above."),
        P("sweep", "object",
          "Strength factors to run the whole spec at, as `{\"strength\": "
          "[0.5, 1.0, 2.0]}`. Every item's `strength` is multiplied by the "
          "factor, and each record gets one row per factor.",
          {"strength": [1.0]}),
        P("control", "bool",
          "Add a factor‑0 run — the model untouched — to the sweep, so every "
          "record has a baseline row (`factor: 0.0`). Set `false` when the "
          "sweep already contains `0` or no baseline is wanted.",
          True),
        P("readout", "object",
          "What to read after the intervened forward pass. "
          "`{\"kind\": \"decision\"}` records the next-token distribution at "
          "the last position. `{\"kind\": \"capture\", \"points\": "
          "[\"blocks.14.resid_post\"], \"position\": \"final\"}` records the "
          "activation vectors at the named hook points instead — `position` "
          "is `\"final\"` or a token index. (A capture readout is for "
          "reading; it is not yet accepted as another item's `source`, which "
          "must be a `residual_vectors` record.) `readout.top_k` overrides "
          "the `top_k` param.",
          {"type": "decision"}),
        P("top_k", "int",
          "How many of the most likely next tokens to record per row in a "
          "decision readout.",
          5),
        P("track", "string",
          "A token whose log-probability is recorded in every decision row "
          "as `track_logp` — the answer you expect the intervention to "
          "promote or suppress. Tokenized as a continuation, so include the "
          "leading space where the model would. A record's own `track` "
          "field takes precedence.",
          None),
        P("outcomes", "list[string]",
          "Tokens whose probabilities are recorded per row as "
          "`outcome_mass` — the candidate answers of a forced choice. A "
          "record's own `outcomes` field takes precedence.",
          None),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$prompts"},
        "spec": [{
            "point": "resid_post", "layers": [14], "positions": "last",
            "op": "add", "strength": 4.0,
            "direction": {"$fetch": "$direction"},
        }],
        "sweep": {"strength": [0.5, 1.0, 2.0]},
        "track": " Paris",
        "top_k": 10,
    },
)

ABLATE_LAYERS = Op(
    name="intervene/layers",
    summary=(
        "Remove one layer's contribution at a time and measure how much the "
        "target token's log-probability drops — which layers the answer "
        "depends on."
    ),
    description="""\
For each record, the block runs the prompt once untouched to get the target
token's baseline log-probability, then once per layer with that layer's
chosen `component` ablated, and records the difference. A more negative
`delta_logp` means the layer was carrying more of the answer.

The ablation replaces the component's output with zero, so the residual
stream passes through that layer unchanged by it.
""",
    inputs=_PROMPT_INPUT,
    emits=Emits('intervene/ablation', collection=True, doc="Per record, one item per layer (`id`, `layer`, `delta_logp`) plus a summary item with `layer: null` carrying `baseline_logp`, `target_token` and `target_id`. The header's `aggregates.mean_delta` and `aggregates.median_delta` are per-layer across all records, in `layers` order."),
    params=(
        P("component", "string",
          "What to remove at each layer: `\"block\"` (the whole layer — "
          "attention and MLP), `\"attention\"`, `\"mlp\"`, or `\"gate\"` (the "
          "per-layer input gate on MatFormer-style models; other "
          "architectures refuse it).",
          "block"),
        _LAYERS_ALL,
        _target("the answer whose dependence on each layer is measured"),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$prompts"},
        "component": "mlp",
        "layers": "all",
        "target": " Paris",
    },
)

ABLATE_HEADS = Op(
    name="intervene/heads",
    summary=(
        "Zero one attention head at a time across the chosen layers and "
        "measure the drop in the target token's log-probability — a "
        "layer × head map of which heads the answer runs through."
    ),
    description="""\
The head-level version of `intervene/layers`. For each record the baseline
log-probability of the target is taken once; then every (layer, head) pair in
turn has that head's output zeroed and the target re-read. The differences
are averaged over records into one matrix.

Cost is one forward pass per record per (layer, head): on a 30-layer,
16-head model that is 480 passes per record, so name the layers you care
about rather than all of them when the prompt set is large.
""",
    inputs=_PROMPT_INPUT,
    emits=Emits('intervene/heads', collection=False, doc="`mean_delta` is a matrix indexed `[layer][head]` of the mean Δ log‑p across records; `conditions` lists each record's `target_token` and `baseline_logp`; `layers` and `n_heads` give the axes."),
    params=(
        _LAYERS_ALL,
        _target("the answer whose dependence on each head is measured"),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$prompts"},
        "layers": [10, 11, 12, 13, 14, 15],
        "target": " Paris",
    },
)

ATTENTION_PATTERNS = Op(
    name="activations/attention",
    summary=(
        "Record the attention weights of every head at the named layers — "
        "which earlier tokens each position attends to."
    ),
    description="""\
One forward pass per record captures the post-softmax attention weights at
each named layer. For each head the result is a square matrix over the
prompt's tokens: row = the attending position, column = the position
attended to.

`layers` must be given explicitly — the matrices are quadratic in prompt
length and there is one per head, so "all layers of a long prompt" is a
picture nobody asked for. The block refuses a capture that would exceed two
million values.
""",
    inputs=_PROMPT_INPUT.replace(" A record may carry its own `target`.", ""),
    emits=Emits('activations/attention', collection=True, doc="One item per record: `tokens` (the prompt's tokens, in order) and `layers` — for each layer, `heads`: a list of `[position][position]` matrices, one per head."),
    params=(
        P("layers", "list[int]",
          "The layers whose attention to record. Must be named — `\"all\"` is "
          "refused."),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$prompts"},
        "layers": [5, 6],
    },
)

ATTRIBUTION_LOGITS = Op(
    name="logits/attribution",
    summary=(
        "Split the target token's final logit into the additive contribution "
        "of the embedding and of every layer — direct logit attribution, with "
        "each row checking that the pieces sum to the truth."
    ),
    description="""\
The residual stream at the last position is the embedding plus each layer's
update. Each of those components is projected through the unembedding (folded
with the final norm's scale when `apply_ln` is on) to give its contribution
to the target's logit, so the contributions are exactly additive.

Every row also reports its **additivity residual**: the summed contributions
minus the model's true final logit for the target. A reader never has to take
the decomposition on faith — a residual far from zero says the decomposition
does not describe this model.

If a record carries a `contrast` token, the contributions are to the
*difference* of the two logits (target minus contrast), which is usually the
more interpretable quantity.

Because additivity only holds over the whole stream, `layers` must be
`"all"`.
""",
    inputs=(
        "`records` — the prompts, one per record (`user`, `prompt` or `text`). "
        "A record may carry `target` and `contrast` tokens."
    ),
    emits=Emits('logits/attribution', collection=True, doc="One item per record: `contributions` in the order the header's `components` names the pieces (`embed`, `L0`, `L1`, …), `target_token`, `contrast_token`, the `additivity` check (`summed`, `true_logit`, `residual`), and `per_head` when `per_head_layers` was set — each listed layer's contribution split by attention head."),
    params=(
        P("apply_ln", "bool",
          "Fold the final norm's scale into the unembedding so contributions "
          "are in the same units as the model's real logits. Turning it off "
          "gives the raw, un-normalised projection.",
          True),
        P("layers", "list[int] | \"all\"",
          "Must be `\"all\"`: the decomposition is only additive over the "
          "whole stream.",
          "all"),
        P("per_head_layers", "list[int]",
          "Layers at which to also split the attention contribution by "
          "head. Opt-in per layer because per-head outputs cost a slower "
          "attention path.",
          None),
        _target("the logit being decomposed"),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$prompts"},
        "target": " Paris",
        "per_head_layers": [12, 13],
    },
)

PATCH_TRACE = Op(
    name="intervene/trace",
    summary=(
        "Causal tracing: run a clean and a corrupted prompt, patch the clean "
        "activations into the corrupted run one (layer, position) at a time, "
        "and map where the clean answer's probability comes back."
    ),
    description="""\
Each record is a pair: a `clean` prompt where the model gets the answer, and
a `corrupt` prompt (same length in tokens) where it does not. The block runs
the clean prompt once, capturing the residual stream at every layer, and the
corrupt prompt once for a baseline. Then for every (layer, position) it runs
the corrupt prompt again with that one activation replaced by the clean
one, and records how much of the target's probability is recovered.

The result is a heat map over (layer, position). Bright cells — where a
single patch restores the answer — are where the fact is carried.

Pairs whose prompts tokenize to different lengths cannot be aligned and are
reported as errors rather than silently shifted.
""",
    inputs=(
        "`records` — pairs, each with `clean` and `corrupt` prompt strings, "
        "and optionally a `target`."
    ),
    emits=Emits('intervene/trace', collection=True, doc="One item per record: `tokens` (of the corrupt prompt), `target_token`, `p_target_clean`, `p_target_corrupt`, and `recovery`, a `[layer][position]` matrix of the change in the target's `metric` relative to the corrupt baseline. A pair that could not be aligned has `error` instead."),
    params=(
        _LAYERS_ALL,
        P("metric", "string",
          "What is recovered: `\"logprob\"` (the target's log-probability — "
          "registers recovery at any probability mass) or `\"prob\"` (raw "
          "probability — only registers when the clean prompt puts real "
          "mass on the target).",
          "logprob"),
        P("point", "string",
          "The residual point patched: `\"resid_post\"` (after the layer) or "
          "`\"resid_pre\"` (before it).",
          "resid_post"),
        _target("the clean answer whose recovery is traced; defaults to the "
                "clean prompt's top‑1"),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$pairs"},
        "target": " Paris",
        "metric": "logprob",
    },
)

RESIDUALS_DIVERGENCE = Op(
    name="activations/divergence",
    summary=(
        "Run two prompts that differ in one place and measure, at every "
        "(layer, position), how far their residual streams have drifted "
        "apart — where a one-token change ripples."
    ),
    description="""\
Each record is a matched pair `a`/`b` that must tokenize to the same length.
Both prompts are run with the residual stream captured at every requested
layer, and for each (layer, position) the block reports 1 − cosine
similarity between the two streams: 0 where they are identical, rising where
the change has propagated.

Unequal-length pairs are reported as errors, not aligned by guesswork.
""",
    inputs="`records` — pairs, each with prompt strings `a` and `b`.",
    emits=Emits('activations/divergence', collection=True, doc='One item per record: `tokens` (from prompt `a`) and `divergence`, a `[layer][position]` matrix of 1 − cosine; an unaligned pair has `error` instead.'),
    params=(
        _LAYERS_ALL,
        P("point", "string",
          "Which residual to compare: `\"post\"` (after each layer) or "
          "`\"pre\"` (before it).",
          "post"),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$pairs"},
        "layers": "all",
    },
)

RESIDUALS_VECTORS = Op(
    name="activations/vectors",
    summary=(
        "Capture the residual-stream vector of each prompt at chosen layers "
        "and a chosen position (or pooled over the sequence) — the raw "
        "material for every geometry measurement."
    ),
    description="""\
One forward pass per record; for each requested layer the block records one
vector. Where in the sequence the vector is read is the important choice:

* `position: "final"` — the last token: right for a prompt whose meaning
  sits at its end (a question awaiting its answer).
* `position: "subject"` — the last token of the record's `subject` string
  within the prompt.
* an integer — that token index.
* `pool` — read the whole sequence and reduce it: `"mean"`, `"max"`,
  `"last_k"` (the last `pool_k` positions) or `"first_k"` (the `pool_k`
  positions after skipping `pool_skip`). Right for a document: embedding a
  story at its final token embeds its *ending*, and a corpus with varied
  endings and identical middles would look varied.

A label rides along on each row (the record's `label`, or the coordinate
named by `label_coord`) so that `geometry/similarity` and `direction/*` can
group rows without parsing ids.

With `source: "queries"` or `"keys"` the block captures attention Q or K
vectors instead of the residual, one row per (layer, head).

The block refuses a capture of more than two million values; use fewer
layers or records.
""",
    inputs=(
        "`records` — the prompts (`user`, `prompt` or `text`), each optionally "
        "with a `label`, `coords`, and a `subject` when `position` is "
        "`\"subject\"`."
    ),
    emits=Emits('activations/vector', collection=True, doc='One item per record per layer: `{id, label, layer, vector}`, plus `head` for Q/K sources and `n_pooled` when pooled. The header carries `point`, `source`, `position` (`"pooled"` when pooled), `layers`, `d_model`, and `skipped_empty` listing any records dropped under `skip_empty`.'),
    params=(
        _LAYERS_ALL,
        P("point", "string",
          "Which residual to read: `\"post\"` (after each layer) or `\"pre\"` "
          "(before it).",
          "post"),
        P("source", "string",
          "What to capture: `\"resid\"` (the residual stream, one vector per "
          "layer), `\"queries\"` or `\"keys\"` (attention Q or K, one vector "
          "per layer per head).",
          "resid"),
        P("position", "\"final\" | \"subject\" | int",
          "Which token's vector to read: `\"final\"` (the last), "
          "`\"subject\"` (the last token of the record's `subject` string), "
          "or an index. Ignored when `pool` is set.",
          "final"),
        P("pool", "string",
          "Read the whole sequence instead of one position and reduce it: "
          "`\"mean\"`, `\"max\"`, `\"last_k\"` or `\"first_k\"`.",
          None),
        P("pool_k", "int",
          "For `last_k`/`first_k`: how many positions the window covers. "
          "Required with those pools.",
          None),
        P("pool_skip", "int",
          "Positions to skip from the start before pooling — a document's "
          "envelope or prompt prefix, so the vector is of its body.",
          0),
        P("label_coord", "string",
          "Which key of a record's `coords` to use as its `label` when the "
          "record has no `label` field — the grouping the geometry "
          "measurements downstream will use.",
          None),
        P("skip_empty", "bool",
          "Drop records with no text instead of refusing. The dropped ids "
          "are reported in `skipped_empty`, because dropping changes n.",
          False),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$stories"},
        "layers": [8, 12, 16],
        "pool": "first_k", "pool_skip": 5, "pool_k": 25,
        "label_coord": "genre",
    },
)

LENS_POSITIONS = Op(
    name="logits/lens",
    summary=(
        "Logit lens over the whole prompt: at every (layer, position), how "
        "probable and how highly ranked the target token is when that "
        "residual is read straight through the unembedding."
    ),
    description="""\
One forward pass per record, capturing the residual after every requested
layer. Each captured vector, at each position, is projected through the
model's unembedding as if it were the final layer, and the target token's
log-probability and rank are read off. Rank 0 means the target is that
position's top readout.

The map answers: where in the sequence, and at what depth, does the answer
become visible?
""",
    inputs=_PROMPT_INPUT,
    emits=Emits('logits/lens', collection=True, doc='One item per record: `tokens`, `target_token`, `target_id`, and two `[layer][position]` matrices: `logprob` and `rank`.'),
    params=(
        _LAYERS_ALL,
        _target("the answer being watched for"),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$prompts"},
        "target": " Paris",
    },
)

LENS_TRAJECTORY = Op(
    name="logits/funnel",
    summary=(
        "Logit lens at the decision point: for each chat-shaped record, the "
        "top-1 token, its probability and the entropy at every layer — the "
        "curve of how a model commits to an answer."
    ),
    description="""\
The record is rendered as a chat (system, user, and an optional prefill that
begins the assistant's turn), run once, and the residual after every layer at
the final position is projected through the unembedding. Per layer the block
records the most likely token, its probability, and the distribution's
entropy in bits.

Read the layers in order and you see the commitment funnel: entropy falling,
one token taking over, at whichever depth this model decides. The output is
a document collection so that a set of trajectories renders as overlaid
curves.
""",
    inputs=(
        "`records` — chat-shaped records with the fields named by "
        "`system_field`, `user_field` and `prefill_field`, and optionally "
        "`coords`."
    ),
    emits=Emits('logits/funnel', collection=True, doc="One item per input record, with `metadata.layers` — a list of `{layer, top1, p, entropy_bits}` — and the record's `coords`."),
    params=(*_FIELD_PARAMS, _PREFILL_FIELD),
    example={
        "model": "$model",
        "records": {"$fetch": "$conditions"},
        "prefill_field": "prefill",
    },
)

STEER_INJECT = Op(
    name="intervene/steer",
    summary=(
        "Build a steering direction from labelled residual vectors (one "
        "label's centroid minus another's), add it to the residual stream "
        "at one layer and position, and sweep its strength while reading the "
        "next-token distribution."
    ),
    description="""\
The direction comes from *data* flowing through the graph: a
`residual_vectors` record whose rows carry labels. At the injection layer,
the block takes the mean vector of the rows labelled `direction.positive`,
subtracts the mean of those labelled `direction.negative`, and adds the
result — scaled by each `alpha` in turn — to the residual stream of every
prompt at the chosen position. Alpha 0 is the built-in control.

For anything beyond one direction at one layer and position, use
`intervene/apply`, of which this is a special case.
""",
    inputs="""\
`records` — the prompts to steer (`user`, `prompt` or `text`); a record may
carry its own `position`, `track` and `tracks`.

`vectors` (by edge, or the `vectors` param) — a `residual_vectors` record
with labelled rows at the injection `layer`.
""",
    emits=Emits('intervene/readout', collection=True, doc="One item per record per alpha: `top` (the most likely next tokens and their log-probs), `track_logp` and `tracks` when asked for. The header's `direction` reports the two labels, the direction's norm and how many vectors went into each centroid; `alphas` lists the sweep."),
    params=(
        P("layer", "int",
          "The layer whose residual stream the direction is added to. Rows "
          "at this layer must exist in the vectors record."),
        P("direction", "object",
          "Which labels define the direction: `{\"positive\": \"formal\", "
          "\"negative\": \"casual\"}` — the centroid of `positive` rows minus "
          "the centroid of `negative` rows."),
        P("alphas", "list[float]",
          "The strengths to sweep. Each prompt is run once per alpha; "
          "alpha `0` is the untouched control.",
          [-8.0, -4.0, 0.0, 4.0, 8.0]),
        P("position", "\"final\" | int",
          "The token position the direction is added at. A record's own "
          "`position` takes precedence.",
          "final"),
        P("top_k", "int",
          "How many of the most likely next tokens to record per row.",
          5),
        P("track", "string",
          "A token whose log-probability is recorded per row as "
          "`track_logp`. A record's own `track` takes precedence.",
          None),
        P("tracks", "object",
          "Several tokens to follow at once, by name: `{\"yes\": \" Yes\", "
          "\"no\": \" No\"}` records each one's log-probability per row under "
          "`tracks`. A record's own `tracks` takes precedence.",
          None),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$prompts"},
        "layer": 12,
        "direction": {"positive": "formal", "negative": "casual"},
        "alphas": [-4.0, 0.0, 4.0, 8.0],
        "track": " certainly",
    },
)

# --- reading and generating --------------------------------------------------------

GENERATE = Op(
    name="text/generate",
    summary=(
        "Sample completions from the model for each chat-shaped record — n "
        "per record, reproducibly seeded — into a document collection."
    ),
    description="""\
Each record is rendered as a chat (system and user turns) and prefilled once;
then `n` completions are sampled from that prefix with the given temperature
and nucleus settings. Every sample's random stream is derived from
(`seed`, record id, sample index), so a sample is a pure function of its key:
running the same node again reproduces the same texts, and growing a corpus
later is the same node over a later `start` range, unioned with the first.

With `fidelity: "trace"` each item also keeps its token ids, character
offsets and the prompt/body segmentation, which is what `score` needs to
annotate it token by token.
""",
    inputs=(
        "`records` — chat-shaped records with the fields named by "
        "`system_field` and `user_field`, an `id`, and optionally `coords`."
    ),
    emits=Emits('text/document', collection=True, doc="`n` items per record, ids `<record id>-s<k>`: `text`, `metadata.coords` (the record's, plus `sample: k`), `metadata.sampling`, and the wire form of the model. At trace fidelity each item also has `trace` (`token_ids`, `text`, `offsets`, `generation_spans`) and `segmentations`. The header carries `fidelity`."),
    params=(
        *_FIELD_PARAMS,
        P("n", "int", "How many completions to sample per record.", 1),
        P("start", "int",
          "The first sample index. Indices run `start` … `start + n − 1`; "
          "a later node with a later `start` extends the corpus without "
          "re-sampling what exists.",
          0),
        P("temperature", "float",
          "Sampling temperature. `0` would be greedy; the default is a "
          "creative-writing setting.",
          0.9),
        P("top_p", "float",
          "Nucleus sampling: only the smallest set of tokens whose "
          "probabilities sum to `top_p` is sampled from.",
          0.95),
        P("max_tokens", "int", "The longest completion, in tokens.", 256),
        P("fidelity", "string",
          "`\"text\"` keeps the completion text; `\"trace\"` also keeps token "
          "ids, offsets and spans, so the collection can be scored token by "
          "token.",
          "text"),
    ),
    example={
        "model": "$model",
        "records": {"$fetch": "$prompts"},
        "n": 4,
        "temperature": 0.9,
        "max_tokens": 200,
        "fidelity": "trace",
    },
)

DECISION_READ = Op(
    name="logits/decision",
    summary=(
        "Read the model's exact next-token distribution at a decision point "
        "— entropy, the top tokens, and the probability mass on each named "
        "outcome — for every chat-shaped record."
    ),
    description="""\
Each record is rendered as a chat and, when a prefill is given, the
assistant's turn is begun with it, so the read happens at the first token
*after* the prefill — `'{ "name": "'` reads the first token of a JSON value.
One prefill pass per record gives the full distribution; nothing is sampled.

`outcomes` turns the read into a forced choice: each outcome string is
tokenized as a continuation and the probability of its first token is
recorded. A `rollout` goes further, expanding the most probable *complete*
outcomes token by token (best-first, reusing the prompt cache) so that
multi-token answers are compared as wholes.

Records keep their `coords`, so a grid of conditions comes out as a grid of
readings.
""",
    inputs=(
        "`conditions` (by edge, or the `conditions` param) — chat-shaped "
        "records with the fields named by `system_field`, `user_field` and "
        "`prefill_field`; a record may carry its own `outcomes`."
    ),
    emits=Emits('logits/decision', collection=True, doc='One item per input record: `id`, `coords`, `entropy_bits`, `top_tokens` (the ten most probable, with `p`), `outcome_mass` (outcome → probability) when outcomes were given, and `rollout` when one was requested.'),
    params=(
        P("conditions", "list[record] | ref",
          "The records to read, when they do not arrive by edge on the "
          "`conditions` port.",
          None),
        P("outcomes", "list[string]",
          "The candidate answers: each is tokenized as a continuation of "
          "the prompt and the probability of its first token is recorded. A "
          "record's own `outcomes` field takes precedence.",
          None),
        P("rollout", "object",
          "Expand complete multi-token outcomes best-first from the "
          "decision point: `{\"top_k\": 10, \"max_tokens\": 8, "
          "\"max_forwards\": 128, \"floor\": 0.001, \"terminators\": "
          "[\"\\\"\"]}` — keep the `top_k` most probable completions, each "
          "at most `max_tokens` long, stopping at a terminator, pruning "
          "branches below `floor`, within a budget of `max_forwards` model "
          "calls.",
          None),
        *_FIELD_PARAMS,
        _PREFILL_FIELD,
    ),
    example={
        "model": "$model",
        "conditions": {"$fetch": "$conditions"},
        "prefill_field": "prefill",
        "outcomes": ["red", "blue", "green"],
    },
)

SCORE = Op(
    name="text/score",
    summary=(
        "Annotate every token of a trace-fidelity collection with its "
        "surprisal in bits under the model — how unexpected each token was."
    ),
    description="""\
For each item the block replays its stored token ids through the model and
records, for every position, −log₂ p(token | everything before it). Prompt
and envelope tokens are scored too, not only the generated body, so the
annotation covers the whole text and a reader can compare.

The collection must have been generated at `fidelity: "trace"`; a text-only
item has no token ids to replay and the block refuses it.
""",
    inputs=(
        "`collection` (by edge) — a trace-fidelity `document_collection`, "
        "usually from `generate`. Or name a stored one with "
        "`collection_path`."
    ),
    emits=Emits('text/annotation', collection=True, doc='One item per token: `{anchor: {item_id, token_start, token_end}, value}` with the surprisal in bits. The header names the `collection` scored and carries `value_type: "numeric"` and `required_fidelity: "trace"`.'),
    params=(
        P("collection_path", "ref",
          "A stored collection to score, when none arrives by edge.",
          None),
    ),
    example={"model": "$model", "collection_path": "$collection"},
)

TOKENIZE_STATS = Op(
    name="text/tokenize",
    summary=(
        "Measure how a tokenizer splits a set of items — as continuations of "
        "a prefix — with a depth histogram, fragmentation, script "
        "composition, and an optional pass/fail gate on expected depth."
    ),
    description="""\
A tokenizer is part of the model, and its behaviour bears on results before
any forward pass: an adapter trained to answer in "one token per slot" only
means something if the vocabulary tokenizes that way *inside its envelope*.
So the block measures items as continuations of a `prefix` — the boundary
matters, because BPE merges across it — which is how training and readout
both see them.

For each item it records the number of tokens it contributes after the
prefix (its **depth**), its pieces, and tokens per whitespace word. Over the
set: the depth histogram, the fraction that are a single token, mean
fragmentation, the Unicode-script composition of the text, and — when
`expect_depth` is given — a gate listing every item that does not tokenize
to exactly that depth.
""",
    inputs="""\
The items to measure come from the first of: the `items` param (strings); a
`vocabulary` (by edge or param — either a list of strings or a training
target map whose weight keys are the items); or `records` (by edge or param),
reading each record's `field`.
""",
    emits=Emits('text/tokenization', collection=False, doc="`n_items`, `mean_depth`, `min_depth`, `max_depth`, `single_token_fraction`, `mean_tokens_per_word`, `fragmented_fraction`, `rows` (the histogram: `{depth, count, share}`), `script_composition`, `boundary_failures` (items that changed the prefix's own tokenization), `gate` (`{expected_depth, pass, n_violations, violations}` or null), `most_fragmented`, and `items` when kept."),
    params=(
        P("prefix", "string",
          "The envelope each item is tokenized after — `'{ \"name\": \"'` "
          "for a JSON value. Empty measures items on their own.",
          ""),
        P("expect_depth", "int | string",
          "Turn on the gate: every item must tokenize to exactly this many "
          "tokens after the prefix. `\"\"` or `\"none\"` (as a run binding) "
          "means no gate.",
          None),
        P("field", "string",
          "When items come from records: which field holds the text.",
          "text"),
        P("vocabulary", "list[string] | object | ref",
          "The items as a list, or a training target map whose weight keys "
          "are the items. Usually arrives by edge on the `vocabulary` port.",
          None),
        P("top_fragmented", "int",
          "How many of the most fragmented items to list under "
          "`most_fragmented`. `0` omits the list.",
          10),
        P("keep_items", "bool",
          "Also emit every item's own measurement (`depth`, `pieces`, "
          "`tokens_per_word`) under `items`.",
          False),
    ),
    example={
        "model": "$model",
        "items": ["red", "blue", "green", "turquoise"],
        "prefix": "{ \"color\": \"",
        "expect_depth": 1,
    },
)

OPS: tuple[Op, ...] = (
    INTERVENE, ABLATE_LAYERS, ABLATE_HEADS, ATTENTION_PATTERNS,
    ATTRIBUTION_LOGITS, PATCH_TRACE, RESIDUALS_DIVERGENCE, RESIDUALS_VECTORS,
    LENS_POSITIONS, LENS_TRAJECTORY, STEER_INJECT,
    GENERATE, DECISION_READ, SCORE, TOKENIZE_STATS,
)
