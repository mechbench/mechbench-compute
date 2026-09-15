"""Ops that run a model's forward pass: reading, editing or steering its
activations, and generating from it.

Shared conventions, stated once here and referred to from the entries:

* A record's **prompt** is its `user`, `prompt` or `text` field (the
  first present). The chat-shaped blocks (`text/generate`, `logits/decision`,
  `logits/funnel`) read `system`, `user` and `prefill`; a record that
  carries the text under another name goes through `records/rename`
  first.
* `template` is how a prompt is tokenized: `"raw"` as plain text,
  `"chat"` wrapped in the model's chat template as one user turn.
* `layers` is a list of layer indices or `"all"`.
* A **target** is a token whose probability the block follows. It is
  tokenized as a continuation, so include the leading space where the
  model would ("` Paris`", not "`Paris`"); a record's own `target`
  field takes precedence over the param.
* Every op here runs on the node's `model` and accepts an `adapter`
  on its port of that name: a LoRA fused on top of the model for this
  node only, on top of any adapters the model reference carries.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Emits, In, Op, P

#: The port every model-running op has: an adapter fused for this node.
ADAPTER = In("adapter", "adapter/lora",
             "A LoRA adapter to fuse on top of the model for this node only — "
             "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
             "reference, or a stored adapter. Fuses last, on top of any "
             "adapters the model reference itself carries; `adapter_scale` "
             "scales this one.",
             required=False)

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


_PROMPTS = In("records", "records/record",
              "The prompts, one per record; a record's prompt is its `user`, "
              "`prompt` or `text` field. A record may carry its own `target`.",
              many=True)

_PAIRS = In("records", "records/pair",
            "Pairs, each with prompt strings `a` and `b`.", many=True)

_CHAT_RECORDS = In("records", "records/record",
                   "Chat-shaped records: `user` (required), `system` and "
                   "`prefill` (optional), an `id`, and optionally `coords`.",
                   many=True)


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
| `direction` | direction | — | The direction for `add`, `project_out`, `clamp`, `rotate` and (optionally) `patch`. May instead arrive on the node's `direction` port, which fills every item that names none. |
| `direction2` | direction | — | The second axis of the plane for `rotate`. |
| `source` | collection | — | A collection of `activations/vector` — or a capture readout from another `intervene/apply` — supplying replacement activations for `mean`, `resample` and `patch`; items are matched to the item's layer (and point). May instead arrive on the node's `source` port. |
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
    inputs=(
        In("records", "records/record",
           "The prompts to run, one forward pass each; a record's prompt is "
           "its `user`, `prompt` or `text` field. A record may also carry its "
           "own `track` and `outcomes`, which take precedence over the params "
           "of the same name.", many=True),
        In("direction", "direction/vector",
           "A direction that fills any spec item without one.", required=False),
        In("source", "activations/vector | intervene/readout",
           "A collection of `activations/vector`, or a capture readout from "
           "another intervention, that fills any `mean`/`resample`/`patch` "
           "item without one.", many=True, required=False),
        ADAPTER,
    ),
    emits=Emits('intervene/readout', collection=True, doc='One item per record per factor: `id`, `coords`, `factor`, and the readout — for a decision, `entropy_bits`, `top` (the most likely next tokens, each `{token, p, logp}`) and `tracked` (name → `{token, p, logp}` for the tokens asked about); for a capture, `position` and `captures`, a collection of `activations/vector` with one item per hook point, each in its own `space` (at most 4096 values). The header carries `spec` (the list as run, with directions and sources replaced by their provenance) and `sweep` (the factors, including `0.0` when a control was added).'),
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
          "is `\"final\"` or a token index. A capture readout is itself "
          "accepted as another intervention's `source`. `readout.top_k` "
          "overrides the `top_k` param.",
          {"type": "decision"}),
        P("top_k", "int",
          "How many of the most likely next tokens to record per row in a "
          "decision readout.",
          5),
        P("tracked", "object",
          "Tokens to follow by name: `{\"yes\": \" Yes\", \"no\": \" No\"}` "
          "records each one's probability and log-probability under "
          "`tracked.<name>`. Tokenized as a continuation, so include the "
          "leading space where the model would. A record's own `tracked` "
          "field takes precedence.",
          None),
        P("track", "string",
          "The older spelling of one tracked token: recorded under "
          "`tracked` by its own text. A record's own `track` field takes "
          "precedence.",
          None),
        P("outcomes", "list[string]",
          "The older spelling of tracked tokens: each outcome is recorded "
          "under `tracked` by its own text — the candidate answers of a "
          "forced choice. A record's own `outcomes` field takes precedence.",
          None),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "spec": [{
            "point": "resid_post", "layers": [14], "positions": "last",
            "op": "add", "strength": 4.0,
            "direction": {"$fetch": "$direction"},
        }],
        "sweep": {"strength": [0.5, 1.0, 2.0]},
        "tracked": {"answer": " Paris"},
        "top_k": 10,
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
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
    inputs=(_PROMPTS, ADAPTER),
    emits=Emits('intervene/ablation', collection=True, doc="Per record, one item per layer: `id`, `layer`, `delta_logp`. The header's `conditions` carry each record's untouched read (`{id, target, baseline_logp}`), and `aggregates.mean_delta` / `aggregates.median_delta` are per-layer across all records, in `layers` order."),
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
        "component": "mlp",
        "layers": "all",
        "target": " Paris",
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
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
    inputs=(_PROMPTS, ADAPTER),
    emits=Emits('intervene/heads', collection=False, doc="One grid over axes `[layer, head]`: `measures.mean_delta` is the mean Δ log‑p across records; `conditions` lists each record's `{id, target, baseline_logp}`; `layers` and `n_heads` give the axes."),
    params=(
        _LAYERS_ALL,
        _target("the answer whose dependence on each head is measured"),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "layers": [10, 11, 12, 13, 14, 15],
        "target": " Paris",
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
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
    inputs=(
        In("records", "records/record",
           "The prompts, one per record; a record's prompt is its `user`, "
           "`prompt` or `text` field.", many=True),
        ADAPTER,
    ),
    emits=Emits('activations/attention', collection=True, doc="One grid per record over axes `[layer, head, query, key]`: `measures.weight` is indexed in that order (row = the attending position, column = the attended-to position); `tokens` are the prompt's tokens."),
    params=(
        P("layers", "list[int]",
          "The layers whose attention to record. Must be named — `\"all\"` is "
          "refused."),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "layers": [5, 6],
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
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
        In("records", "records/record",
           "The prompts, one per record (`user`, `prompt` or `text`). A "
           "record may carry `target` and `contrast` tokens.", many=True),
        ADAPTER,
    ),
    emits=Emits('logits/attribution', collection=True, doc="One grid per record over the axis `[component]`, in the order the header's `components` names the pieces (`embed`, `L0`, `L1`, …): `measures.contribution`, the `target` and `contrast` tokens, the `additivity` check (`summed`, `true_logit`, `residual`), and `per_head` when `per_head_layers` was set — each listed layer's contribution split by attention head."),
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
        "target": " Paris",
        "per_head_layers": [12, 13],
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
)

PATCH_TRACE = Op(
    name="intervene/trace",
    summary=(
        "Causal tracing: run a clean and a corrupted prompt, patch the clean "
        "activations into the corrupted run one (layer, position) at a time, "
        "and map where the clean answer's probability comes back."
    ),
    description="""\
Each record is a pair: prompt `a` (clean) where the model gets the answer,
and prompt `b` (corrupted, the same length in tokens) where it does not — the
older field names `clean` and `corrupt` are still read. The block runs
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
        In("records", "records/pair",
           "Pairs, each with prompt strings `a` and `b`, and optionally a "
           "`target`.", many=True),
        ADAPTER,
    ),
    emits=Emits('intervene/trace', collection=True, doc="One grid per record over axes `[layer, position]`: `measures.recovery` is the change in the target's `metric` from the `b` baseline when the `a` residual is patched in; `tokens` are prompt `b`'s; `target`, `metric`, `value_a` and `value_b` (the metric on each prompt) ride along. A pair that could not be aligned has `error` and empty measures."),
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
        "target": " Paris",
        "metric": "logprob",
    },
    example_inputs={"records": {"$fetch": "$pairs"}},
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
    inputs=(_PAIRS, ADAPTER),
    emits=Emits('activations/divergence', collection=True, doc="One grid per record over axes `[layer, position]`: `measures.divergence` is 1 − cosine; `tokens` are prompt `a`'s. An unaligned pair has `error` and empty measures."),
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
        "layers": "all",
    },
    example_inputs={"records": {"$fetch": "$pairs"}},
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

Every item carries its `space` (`{model, layer, point, head?, d}`) and the
record's `coords`, which is what `geometry/similarity`, `geometry/mst` and
`direction/*` group on (their `axis` names the coordinate).

With `source: "queries"` or `"keys"` the block captures attention Q or K
vectors instead of the residual, one row per (layer, head).

The block refuses a capture of more than two million values; use fewer
layers or records.
""",
    inputs=(
        In("records", "records/record",
           "The prompts (`user`, `prompt` or `text`), each optionally with "
           "`coords`, and a `subject` when `position` is `\"subject\"`. A "
           "document collection is read the same way.", many=True),
        ADAPTER,
    ),
    emits=Emits('activations/vector', collection=True, doc='One item per record per layer (per head, for Q/K sources): `{id, coords, space, vector, norm}`, plus `token` (the token read, when not pooled) and `n_pooled` when pooled. The header carries `model`, `point`, `source`, `position` (`"pooled"` when pooled), `layers`, `d_model`, and `skipped_empty` listing any records dropped under `skip_empty`.'),
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
        P("skip_empty", "bool",
          "Drop records with no text instead of refusing. The dropped ids "
          "are reported in `skipped_empty`, because dropping changes n.",
          False),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "layers": [8, 12, 16],
        "pool": "first_k", "pool_skip": 5, "pool_k": 25,
    },
    example_inputs={"records": {"$fetch": "$stories"}},
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
    inputs=(_PROMPTS, ADAPTER),
    emits=Emits('logits/lens', collection=True, doc="One grid per record over axes `[layer, position]`: `measures.logprob` and `measures.rank` (0 is the top readout), `tokens`, and the `target` token."),
    params=(
        _LAYERS_ALL,
        _target("the answer being watched for"),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "target": " Paris",
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
)

LENS_TRAJECTORY = Op(
    name="logits/funnel",
    summary=(
        "Logit lens at the decision point: for each chat-shaped record, the "
        "top-1 token, its probability and the entropy at every layer — the "
        "curve of how a model commits to an answer."
    ),
    description="""\
The record is rendered as a chat (`system`, `user`, and an optional
`prefill` that begins the assistant's turn), run once, and the residual
after every layer at the final position is projected through the
unembedding. Per layer the block records the most likely token, its
probability, and the distribution's entropy in bits.

Read a record's items in layer order and you see the commitment funnel:
entropy falling, one token taking over, at whichever depth this model
decides. A set of records renders as overlaid curves.
""",
    inputs=(_CHAT_RECORDS, ADAPTER),
    emits=Emits('logits/funnel', collection=True, doc="One item per record per layer: `id`, `coords`, `layer`, and the distribution read through the unembedding at that layer — `entropy_bits`, `top` (the `top_k` most likely tokens, each `{token, p, logp}`) and `tracked`. The header carries `layers` and `top_k`."),
    params=(
        P("top_k", "int", "How many of the most likely tokens to record per layer.", 5),
        P("tracked", "object",
          "Tokens to follow by name: `{\"yes\": \" Yes\", \"no\": \" No\"}` "
          "records each one's probability and log-probability under "
          "`tracked.<name>`. Tokenized as a continuation, so include the "
          "leading space where the model would. A record's own `tracked` "
          "field takes precedence.",
          None),
    ),
    example={
        "model": "$model",
        "top_k": 5,
    },
    example_inputs={"records": {"$fetch": "$conditions"}},
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
The direction comes from *data* flowing through the graph: a collection of
`activations/vector` whose items are grouped on a coordinate. At the
injection layer, the block takes the mean vector of the items whose
`direction.axis` coordinate is `direction.positive`, subtracts the mean of
those at `direction.negative`, and adds the result — scaled by each `alpha`
in turn — to the residual stream of every prompt at the chosen position.
Alpha 0 is the built-in control.

For anything beyond one direction at one layer and position, use
`intervene/apply`, of which this is a special case.
""",
    inputs=(
        In("records", "records/record",
           "The prompts to steer (`user`, `prompt` or `text`); a record may "
           "carry its own `position` and `tracked`.", many=True),
        In("vectors", "activations/vector",
           "The labelled vectors the direction is built from, with items at "
           "the injection `layer`.", many=True),
        ADAPTER,
    ),
    emits=Emits('intervene/readout', collection=True, doc="One item per record per alpha: `id`, `coords`, `factor` (the alpha), `entropy_bits`, `top` (the most likely next tokens, each `{token, p, logp}`) and `tracked`. The header's `direction` reports the axis and the two values, the direction's norm and how many vectors went into each centroid; `sweep` lists the alphas."),
    params=(
        P("layer", "int",
          "The layer whose residual stream the direction is added to. Items "
          "at this layer must exist in the vectors collection."),
        P("direction", "object",
          "Which groups define the direction: `{\"axis\": \"register\", "
          "\"positive\": \"formal\", \"negative\": \"casual\"}` — the "
          "centroid of the items whose `axis` coordinate is `positive` minus "
          "the centroid of those at `negative`. `axis` defaults to `label`."),
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
        P("tracked", "object",
          "Tokens to follow by name: `{\"yes\": \" Yes\", \"no\": \" No\"}` "
          "records each one's probability and log-probability under "
          "`tracked.<name>`. Tokenized as a continuation, so include the "
          "leading space where the model would. A record's own `tracked` "
          "field takes precedence.",
          None),
        P("track", "string",
          "The older spelling of one tracked token: recorded under "
          "`tracked` by its own text. A record's own `track` takes precedence.",
          None),
        P("tracks", "object",
          "The older spelling of `tracked`, read the same way. A record's own "
          "`tracks` takes precedence.",
          None),
        _TEMPLATE,
    ),
    example={
        "model": "$model",
        "layer": 12,
        "direction": {"axis": "register", "positive": "formal", "negative": "casual"},
        "alphas": [-4.0, 0.0, 4.0, 8.0],
        "tracked": {"hedge": " certainly"},
    },
    example_inputs={"records": {"$fetch": "$prompts"}, "vectors": {"$fetch": "$vectors"}},
)

# --- reading and generating --------------------------------------------------------

GENERATE = Op(
    name="text/generate",
    summary=(
        "Sample completions from the model for each chat-shaped record — n "
        "per record, reproducibly seeded — into a document collection."
    ),
    description="""\
Each record is rendered as a chat (`system` and `user` turns) and prefilled
once; then `n` completions are sampled from that prefix with the given
temperature and nucleus settings. Every sample's random stream is derived
from (`seed`, record id, sample index), so a sample is a pure function of
its key: running the same node again reproduces the same texts, and growing
a corpus later is the same node over a later `start` range, unioned with
the first.

With `fidelity: "trace"` each item also keeps its token ids, character
offsets and the prompt/body segmentation, which is what `score` needs to
annotate it token by token.
""",
    inputs=(
        In("records", "records/record",
           "Chat-shaped records: `user` (required) and `system` (optional), "
           "an `id`, and optionally `coords`.", many=True),
        ADAPTER,
    ),
    emits=Emits('text/document', collection=True, doc="`n` items per record, ids `<record id>-s<k>`: `text`, `coords` (the record's, plus `sample: k`), `metadata.sampling`, and the wire form of the model. At trace fidelity each item also has `trace` (`token_ids`, `text`, `offsets`, `generation_spans`) and `segmentations`. The header carries `fidelity`."),
    params=(
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
        "n": 4,
        "temperature": 0.9,
        "max_tokens": 200,
        "fidelity": "trace",
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
)

DECISION_READ = Op(
    name="logits/decision",
    summary=(
        "Read the model's exact next-token distribution at a decision point "
        "— entropy, the top tokens, and the probability mass on each named "
        "outcome — for every chat-shaped record."
    ),
    description="""\
Each record is rendered as a chat and, when it carries a `prefill`, the
assistant's turn is begun with it, so the read happens at the first token
*after* the prefill — `'{ "name": "'` reads the first token of a JSON value.
One prefill pass per record gives the full distribution; nothing is sampled.

`outcomes` turns the read into a forced choice: each outcome string is
tokenized as a continuation and the probability of its first token is
recorded under `tracked` by the outcome's own text. A `rollout` goes
further, expanding the most probable *complete* outcomes token by token
(best-first, reusing the prompt cache) so that multi-token answers are
compared as wholes.

Records keep their `coords`, so a grid of conditions comes out as a grid of
readings.
""",
    inputs=(
        In("conditions", "records/record",
           "Chat-shaped records: `user` (required), `system` and `prefill` "
           "(optional), an `id`, and optionally `coords`; a record may carry "
           "its own `outcomes`.", many=True),
        ADAPTER,
    ),
    emits=Emits('logits/decision', collection=True, doc='One item per input record: `id`, `coords`, `entropy_bits`, `top` (the `top_k` most probable tokens, each `{token, p, logp}`), `tracked` (each outcome and tracked token by name, `{token, p, logp}`), and `rollout` when one was requested. The header carries `top_k`.'),
    params=(
        P("outcomes", "list[string]",
          "The candidate answers: each is tokenized as a continuation of "
          "the prompt and the probability of its first token is recorded "
          "under `tracked` by the outcome's own text. A record's own "
          "`outcomes` field takes precedence.",
          None),
        P("tracked", "object",
          "Other tokens to follow by name, `{\"yes\": \" Yes\"}`, recorded "
          "under `tracked`. A record's own `tracked` takes precedence.",
          None),
        P("top_k", "int", "How many of the most likely tokens to record.", 10),
        P("rollout", "object",
          "Expand complete multi-token outcomes best-first from the "
          "decision point: `{\"top_k\": 10, \"max_tokens\": 8, "
          "\"max_forwards\": 128, \"floor\": 0.001, \"terminators\": "
          "[\"\\\"\"]}` — keep the `top_k` most probable completions, each "
          "at most `max_tokens` long, stopping at a terminator, pruning "
          "branches below `floor`, within a budget of `max_forwards` model "
          "calls.",
          None),
    ),
    example={
        "model": "$model",
        "outcomes": ["red", "blue", "green"],
    },
    example_inputs={"conditions": {"$fetch": "$conditions"}},
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
        In("collection", "text/document",
           "A trace-fidelity document collection, usually from `text/generate`; "
           "a stored one arrives as `{\"$fetch\": …}`.", many=True),
        ADAPTER,
    ),
    emits=Emits('text/annotation', collection=True, doc='One item per token: `{anchor: {item_id, token_start, token_end}, value}` with the surprisal in bits. The header names the `collection` scored and carries `value_type: "numeric"` and `required_fidelity: "trace"`.'),
    params=(),
    example={"model": "$model"},
    example_inputs={"collection": {"$fetch": "$collection"}},
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
    inputs=(
        In("vocabulary", "text/word-list",
           "The items to measure: a word list (`words`), or a frequency table "
           "or training target whose `weights` keys are the items. A bare "
           "list of strings inline is read as the words. One of `vocabulary` "
           "and `records` is required.", required=False),
        In("records", "records/record",
           "Records whose `text` (else `user` or `prompt`) is measured, when "
           "no `vocabulary` is given.", many=True, required=False),
    ),
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
        "prefix": "{ \"color\": \"",
        "expect_depth": 1,
    },
    example_inputs={"vocabulary": ["red", "blue", "green", "turquoise"]},
)

OPS: tuple[Op, ...] = (
    INTERVENE, ABLATE_LAYERS, ABLATE_HEADS, ATTENTION_PATTERNS,
    ATTRIBUTION_LOGITS, PATCH_TRACE, RESIDUALS_DIVERGENCE, RESIDUALS_VECTORS,
    LENS_POSITIONS, LENS_TRAJECTORY, STEER_INJECT,
    GENERATE, DECISION_READ, SCORE, TOKENIZE_STATS,
)
