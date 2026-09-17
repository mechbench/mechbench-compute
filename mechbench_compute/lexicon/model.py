"""Ops that run a model's forward pass: reading, editing or steering its
activations, and generating from it.

Shared conventions, stated once here and referred to from the entries:

* **One rendering.** Every op renders a record the same way. A
  condition — `user`, optional `system`, optional `prefill` — goes
  through the model's chat template as one user turn, the assistant's
  turn begun with the prefill: that is where a decision is read, and
  every op here can read there. A record that carries only `text` or
  `prompt`, or one that says `template: false`, is tokenized raw; a
  record that carries the text under another name goes through
  `records/rename` first.
* **`tracked`** `{name: token}` names the tokens an op reports on; a
  record's own `tracked` takes precedence over the param. The first
  entry is the target — the token a sweep's Δ log p is taken on; with
  none named, the model's own top-1 for that prompt is. Each token is
  tokenized as a continuation, so include the leading space where the
  model would ("` Paris`", not "`Paris`").
* **Positions.** One selector wherever a position is chosen:
  `"last"`, `"all"`, a list of indices (negative from the end),
  `{"tokens": [...]}`, `{"range": [a, b]}`, `{"after": n}`,
  `"subject"` (the last token of the record's `subject` string) or
  `"generated"` (from where generation began). A single-position param
  must resolve to one position. One pooling clause, `pool: {"reduce":
  "mean" | "max", "over": <selector>}`, reads the selected positions
  and reduces them.
* **Points.** `resid_pre`, `resid_post`, `attn_out`, `mlp_out`,
  `gate_out`, `attn.q`, …, `embed`, `final_norm`, `logits` — the one
  vocabulary every `point` param and every `space.point` uses.
* `layers` is a list of layer indices or `"all"`.
* Every op here runs on the node's `model` and accepts an `adapter`
  on its port of that name: a LoRA fused on top of the model for this
  node only, on top of any adapters the model reference carries.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Emits, In, Op, P
from mechbench_compute.lexicon.common import (
    TARGET_TRANSFORM,
    TARGET_UNIFORM,
    TARGET_WEIGHTS,
)

#: The port every model-running op has: an adapter fused for this node.
ADAPTER = In("adapter", "adapter/lora",
             "A LoRA adapter to fuse on top of the model for this node only — "
             "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
             "reference, or a stored adapter. Fuses last, on top of any "
             "adapters the model reference itself carries; `adapter_scale` "
             "scales this one.",
             required=False)

_LAYERS_ALL = P("layers", "list[int] | \"all\"",
                "Which layers to run over.",
                "all")

_POSITIONS_DOC = (
    "`\"last\"`, `\"all\"`, a list of indices (negative from the end), "
    "`{\"tokens\": [...]}`, `{\"range\": [a, b]}`, `{\"after\": n}`, "
    "`\"subject\"` or `\"generated\"`"
)

_RESIDUAL_POINT = P("point", "string",
                    "Which residual stream to read: `\"resid_post\"` (after each "
                    "layer) or `\"resid_pre\"` (before it).",
                    "resid_post", choices=("resid_post", "resid_pre"), value="point")


def _tracked(what: str) -> P:
    return P("tracked", "map[string, string]",
             f"Tokens to follow by name, `{{\"answer\": \" Paris\"}}`; the first "
             f"is the target — {what}. Each is tokenized as a continuation "
             "(include the leading space). A record's own `tracked` takes "
             "precedence; with none named, the model's own top-1 prediction "
             "for that prompt is the target.",
             None)


_ONE_OR_MORE = "int | list[int]"

#: One item of an intervention: an edit at a point in the forward pass,
#: or — naming `parameter` instead — an edit to a weight.
_SPEC_FIELDS = (
    P("point", "string", "Where in the forward pass to act.", "resid_post", value="point"),
    P("parameter", "string",
      "Edit this weight instead of an activation, named as the module tree names it; "
      "`*` stands for one segment.",
      None),
    P("layers", "int | list[int] | \"all\"", "Which layers the item applies to.", "all"),
    P("positions", "selector", "Which token positions.", "last"),
    P("heads", _ONE_OR_MORE, "Only these attention heads, at a point with a head axis.", None),
    P("neurons", _ONE_OR_MORE, "Only these indices along the feature axis.", None),
    P("op", "string", "What to do there — the table above lists each op and what it needs.", "zero",
      choices=("zero", "mean", "resample", "patch", "add", "scale", "clamp", "project_out",
               "rotate", "truncate")),
    P("strength", "float", "The item's magnitude, multiplied by each sweep factor.", 1.0),
    P("direction", "json",
      "The direction, usually `{\"$fetch\": …}`; or it arrives on the node's `direction` port.", None),
    P("direction2", "json", "For `rotate`: the second axis of the plane.", None),
    P("source", "json",
      "For `mean`, `resample` and `patch`: the replacement activations, or they arrive on the "
      "node's `source` port.",
      None),
    P("row", "object", "For `patch`: which row of `source` to write in.", None,
      fields=(P("index", "int", "The row's index.", 0),)),
    P("condition", "object",
      "Act only where the activation projects onto a direction above (or below) a threshold.", None,
      fields=(
          P("direction", "json", "The direction projected onto."),
          P("threshold", "float", "The projection's threshold.", 0.0),
          P("above", "bool", "Act above the threshold; `false` acts below it.", True),
      )),
    P("seed", "int", "The seed `resample` draws with; the node's `seed` by default.", None),
    P("side", "string",
      "For a weight's `project_out`: the side facing the residual stream, where the module's "
      "name does not imply it.",
      None, choices=("in", "out")),
    P("rank", "int", "For a weight's `truncate`: how many singular directions to keep.", None),
)

_PROMPTS = In("records", "records/record",
              "The prompts, one per record; a record's prompt is its `user`, "
              "`prompt` or `text` field. A record may carry its own `tracked`.",
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
    requires="mlx-local",
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
| `positions` | selector | `"last"` | Which token positions: `"last"`, `"all"`, a list of indices (negative from the end), `{"tokens": ["lighthouse"]}` (every position whose token matches), `{"range": [2, 6]}`, `{"after": n}`, `"subject"` or `"generated"`. |
| `heads` | list[int] | all | Restrict the item to these attention heads, at a point that has a head axis (`attn.q`, `attn.k`, `attn.v`, `attn.scores`, `attn.weights`, `attn.per_head_out`, and the pre-norm/pre-rope variants). |
| `neurons` | list[int] | all | Restrict the item to these indices along the feature axis — MLP neurons at `mlp.act`, residual dimensions at `resid_post`, vocabulary entries at `logits`. |
| `op` | string | `"zero"` | What to do there — see the table below. |
| `strength` | float | `1.0` | The item's magnitude: the coefficient for `add`, the factor for `scale`, the bound for `clamp`, the angle in radians for `rotate`. Multiplied by each sweep factor. |
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
| `clamp` | Clip its component along `direction` into [−\|`strength`\|, \|`strength`\|]. | `direction` |
| `project_out` | Remove its component along `direction`. | `direction` |
| `rotate` | Rotate it by `strength` radians in the plane of `direction` and `direction2`. | `direction`, `direction2` |

The older `intervene/ablate-layers` and `intervene/ablate-heads` and `intervene/steer` operations are special cases of this
grammar.

### Editing a weight instead of an activation

A spec item that names a **`parameter`** rather than a `point` edits the
model itself — `{"parameter": "layers.12.self_attn.o_proj", "op":
"project_out", "direction": {"$fetch": "$axis"}}`. The two kinds compose
in one spec, and differ in scope: an activation edit lasts for one
forward pass, a weight edit for the node. The tensor is changed, every
record runs against the changed model, and the original is reinstalled
afterwards — the tensor itself, kept and put back, never a subtraction
that would not round-trip in bf16.

A sweep re-applies the weight edits at each factor, so `{"strength":
[0.5, 1.0]}` on a `project_out` removes half the direction and then all
of it. Factor 0 is the unedited model, which is the control the readout
compares against.

| Weight `op` | Effect | Needs |
|---|---|---|
| `zero` | Set the tensor to zero — the module stops contributing. | — |
| `scale` | Multiply it by `strength`. | — |
| `project_out` | Remove a direction from the side of the matrix that faces the residual stream: what the module WRITES (`o_proj`, `down_proj`) or what it READS (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`). `strength` 1.0 removes it entirely. `side` names the side where it is not implied. | `direction` |
| `truncate` | Keep the top `rank` singular directions and drop the rest — how much of the module survives being low-rank. | `rank` |

`parameter` takes the same names `weights/capture` does, `*` included:
one item can zero every layer's `o_proj`.
""",
    inputs=(
        In("records", "records/record",
           "The prompts to run, one forward pass each; a record's prompt is "
           "its `user`, `prompt` or `text` field. A record may also carry its "
           "own `tracked`, which takes precedence over the param of the same "
           "name.", many=True),
        In("direction", "direction/vector",
           "A direction that fills any spec item without one.", required=False),
        In("source", "activations/vector | intervene/readout",
           "A collection of `activations/vector`, or a capture readout from "
           "another intervention, that fills any `mean`/`resample`/`patch` "
           "item without one.", many=True, required=False),
        ADAPTER,
    ),
    emits=Emits('intervene/readout', collection=True, doc='One item per record per factor: `id`, `coords`, `factor`, and the readout — for a decision, `entropy_bits`, `top` (the most likely next tokens, each `{token, p, logp}`) and `tracked` (name → `{token, p, logp}` for the tokens asked about); for a capture, `position` and `captures`, a collection of `activations/vector` with one item per hook point, each in its own `space` (at most 4096 values). The header carries `spec` (the list as run, with directions and sources replaced by their provenance), `weights` (the parameter edits, when any — so a reader knows the model was not the one on the shelf) and `sweep` (the factors, including `0.0` when a control was added).'),
    params=(
        P("spec", "list[object]",
          "The intervention items, applied together in one forward pass per "
          "record. At least one is required; the fields are described under "
          "*Spec items* above.", fields=_SPEC_FIELDS),
        P("sweep", "object",
          "Strength factors to run the whole spec at, as `{\"strength\": "
          "[0.5, 1.0, 2.0]}`. Every item's `strength` is multiplied by the "
          "factor, and each record gets one row per factor.",
          {"strength": [1.0]}, fields=(
              P("strength", "list[float]", "The factors; `0` is the untouched model.", [1.0]),
          )),
        P("control", "bool",
          "Add a factor‑0 run — the model untouched — to the sweep, so every "
          "record has a baseline row (`factor: 0.0`). Set `false` when the "
          "sweep already contains `0` or no baseline is wanted.",
          True),
        P("readout", "object",
          "What to read after the intervened forward pass. "
          "`{\"type\": \"decision\"}` records the next-token distribution at "
          "the last position. `{\"type\": \"capture\", \"points\": "
          "[\"blocks.14.resid_post\"], \"position\": \"last\"}` records the "
          "activation vectors at the named hook points instead — `position` "
          "is a selector naming one position. A capture readout is itself "
          "accepted as another intervention's `source`. `readout.top_k` "
          "overrides the `top_k` param.",
          {"type": "decision"}, fields=(
              P("type", "string", "`decision` reads the next-token distribution; `capture` reads activations.",
                "decision", choices=("decision", "capture")),
              P("kind", "string", "The older spelling of `type`; read when `type` is absent.", None,
                choices=("decision", "capture")),
              P("top_k", "int", "For `decision`: overrides the node's `top_k`.", None),
              P("points", "list[string]",
                "For `capture`: the hook points to read, such as `blocks.14.resid_post`.", None),
              P("position", "selector", "For `capture`: the one position to read.", "last"),
          )),
        P("top_k", "int",
          "How many of the most likely next tokens to record per row in a "
          "decision readout.",
          5),
        P("tracked", "map[string, string]",
          "Tokens to follow by name: `{\"yes\": \" Yes\", \"no\": \" No\"}` "
          "records each one's probability and log-probability under "
          "`tracked.<name>`. Tokenized as a continuation, so include the "
          "leading space where the model would. A record's own `tracked` "
          "field takes precedence.",
          None),
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
    name="intervene/ablate-layers",
    requires="mlx-local",
    summary=(
        "Remove one layer's contribution at a time and measure how much the "
        "target token's log-probability drops — which layers the answer "
        "depends on."
    ),
    description="""\
For each record, the block runs the prompt once untouched to get the target
token's baseline log-probability, then once per layer with the named
`point`(s) of that layer zeroed, and records the difference. A more negative
`delta_logp` means the layer was carrying more of the answer.

Zeroing a sub-layer's output leaves the residual stream unchanged by it;
zeroing both `attn_out` and `mlp_out` — the default — removes the whole
layer's contribution. Because records render the chat-shaped way, the
sweep can be taken at a decision point inside an assistant turn (a record
with a `prefill`), where `logits/read` reads.
""",
    inputs=(_PROMPTS, ADAPTER),
    emits=Emits('intervene/ablation', collection=True, doc="Per record, one item per layer: `id`, `layer`, `delta_logp`. The header's `conditions` carry each record's untouched read (`{id, target, baseline_logp}`), and `aggregates.mean_delta` / `aggregates.median_delta` are per-layer across all records, in `layers` order."),
    params=(
        P("point", "string | list[string]",
          "The sub-layer output(s) zeroed at each layer: `\"attn_out\"`, "
          "`\"mlp_out\"`, or `\"gate_out\"` (the per-layer input gate on "
          "MatFormer-style models; other architectures refuse it), one or "
          "several. The default zeroes attention and MLP together — the "
          "whole layer.",
          ["attn_out", "mlp_out"], choices=("attn_out", "mlp_out", "gate_out"), value="point"),
        _LAYERS_ALL,
        _tracked("the answer whose dependence on each layer is measured"),
    ),
    example={
        "model": "$model",
        "point": "mlp_out",
        "layers": "all",
        "tracked": {"answer": " Paris"},
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
)

ABLATE_HEADS = Op(
    name="intervene/ablate-heads",
    requires="mlx-local",
    summary=(
        "Zero one attention head at a time across the chosen layers and "
        "measure the drop in the target token's log-probability — a "
        "layer × head map of which heads the answer runs through."
    ),
    description="""\
The head-level version of `intervene/ablate-layers`. For each record the baseline
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
        _tracked("the answer whose dependence on each head is measured"),
    ),
    example={
        "model": "$model",
        "layers": [10, 11, 12, 13, 14, 15],
        "tracked": {"answer": " Paris"},
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
)

ATTENTION_PATTERNS = Op(
    name="activations/capture-attention",
    requires="mlx-local",
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
    ),
    example={
        "model": "$model",
        "layers": [5, 6],
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
)

ATTRIBUTION_LOGITS = Op(
    name="logits/attribute",
    requires="mlx-local",
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

With two `tracked` tokens, the contributions are to the *difference* of
their logits (the first minus the second), which is usually the more
interpretable quantity.

Because additivity only holds over the whole stream, `layers` must be
`"all"`.
""",
    inputs=(
        In("records", "records/record",
           "The prompts, one per record (`user`, `prompt` or `text`). A "
           "record may carry its own `tracked`.", many=True),
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
        _tracked("the logit being decomposed"),
    ),
    example={
        "model": "$model",
        "tracked": {"answer": " Paris"},
        "per_head_layers": [12, 13],
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
)

PATCH_TRACE = Op(
    name="intervene/patch",
    requires="mlx-local",
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
           "Pairs, each with prompt strings `a` and `b`, and optionally its "
           "own `tracked`.", many=True),
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
          "logprob", choices=("logprob", "prob")),
        P("point", "string",
          "The residual point patched: `\"resid_post\"` (after the layer) or "
          "`\"resid_pre\"` (before it).",
          "resid_post", choices=("resid_post", "resid_pre"), value="point"),
        _tracked("the clean answer whose recovery is traced; defaults to the "
                "clean prompt's top‑1"),
    ),
    example={
        "model": "$model",
        "tracked": {"answer": " Paris"},
        "metric": "logprob",
    },
    example_inputs={"records": {"$fetch": "$pairs"}},
)

RESIDUALS_DIVERGENCE = Op(
    name="activations/contrast",
    requires="mlx-local",
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
        _RESIDUAL_POINT,
    ),
    example={
        "model": "$model",
        "layers": "all",
    },
    example_inputs={"records": {"$fetch": "$pairs"}},
)

RESIDUALS_VECTORS = Op(
    name="activations/capture",
    requires="mlx-local",
    summary=(
        "Capture the residual-stream vector of each prompt at chosen layers "
        "and a chosen position (or pooled over the sequence) — the raw "
        "material for every geometry measurement."
    ),
    description="""\
One forward pass per record; for each requested layer the block records one
vector. Where in the sequence the vector is read is the important choice:

* `position: "last"` — the last token: right for a prompt whose meaning
  sits at its end (a question awaiting its answer).
* `position: "subject"` — the last token of the record's `subject` string
  within the prompt; a list of one index names that token.
* `pool` — read a set of positions and reduce them: `{"reduce": "mean",
  "over": "all"}` is the whole sequence, `{"reduce": "mean", "over":
  {"range": [5, 30]}}` a window. Right for a document: embedding a story
  at its last token embeds its *ending*, and a corpus with varied endings
  and identical middles would look varied.

Every item carries its `space` (`{model, layer, point, head?, d}`) and the
record's `coords`, which is what `geometry/compare`, `geometry/span` and
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
        _RESIDUAL_POINT,
        P("source", "string",
          "What to capture: `\"resid\"` (the residual stream, one vector per "
          "layer), `\"queries\"` or `\"keys\"` (attention Q or K, one vector "
          "per layer per head).",
          "resid", choices=("resid", "queries", "keys")),
        P("position", "selector",
          f"Which token's vector to read: {_POSITIONS_DOC}, resolving to a "
          "single position. Ignored when `pool` is set.",
          "last"),
        P("pool", "object",
          "Read a set of positions and reduce them to one vector: "
          "`{\"reduce\": \"mean\" | \"max\", \"over\": <selector>}` — "
          "`{\"reduce\": \"mean\", \"over\": {\"range\": [5, 30]}}` is the "
          "mean over positions 5 … 29 (a document's body after its "
          "envelope); `\"over\": \"all\"` is the whole sequence.",
          None, value="pool"),
        P("skip_empty", "bool",
          "Drop records with no text instead of refusing. The dropped ids "
          "are reported in `skipped_empty`, because dropping changes n.",
          False),
    ),
    example={
        "model": "$model",
        "layers": [8, 12, 16],
        "pool": {"reduce": "mean", "over": {"range": [5, 30]}},
    },
    example_inputs={"records": {"$fetch": "$stories"}},
)

LENS_POSITIONS = Op(
    name="logits/scan",
    requires="mlx-local",
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
        _tracked("the answer being watched for"),
    ),
    example={
        "model": "$model",
        "tracked": {"answer": " Paris"},
    },
    example_inputs={"records": {"$fetch": "$prompts"}},
)

LENS_TRAJECTORY = Op(
    name="logits/read-layers",
    requires="mlx-local",
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
        P("tracked", "map[string, string]",
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
    requires="mlx-local",
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
          "the centroid of those at `negative`. `axis` defaults to `label`.", fields=(
              P("axis", "string", "The coordinate the groups are read on; `label` reads the older field too.",
                "label"),
              P("positive", "string | float", "The `axis` value of the items the direction points toward."),
              P("negative", "string | float", "The `axis` value of the items it points away from."),
          )),
        P("alphas", "list[float]",
          "The strengths to sweep. Each prompt is run once per alpha; "
          "alpha `0` is the untouched control.",
          [-8.0, -4.0, 0.0, 4.0, 8.0]),
        P("position", "selector",
          f"The one token position the direction is added at: {_POSITIONS_DOC}, "
          "resolving to a single position. A record's own `position` takes "
          "precedence.",
          "last"),
        P("top_k", "int",
          "How many of the most likely next tokens to record per row.",
          5),
        P("tracked", "map[string, string]",
          "Tokens to follow by name: `{\"yes\": \" Yes\", \"no\": \" No\"}` "
          "records each one's probability and log-probability under "
          "`tracked.<name>`. Tokenized as a continuation, so include the "
          "leading space where the model would. A record's own `tracked` "
          "field takes precedence.",
          None),
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
    requires="mlx-local",
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
          "text", choices=("text", "trace")),
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
    name="logits/read",
    requires="mlx-local",
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

`tracked` turns the read into a forced choice: each named token is
tokenized as a continuation of the rendered prompt and the probability of
its first token is recorded under `tracked` by its name — `{"1": "1", …,
"6": "6"}` for a die. A `rollout` goes further, expanding the most probable
*complete* outcomes token by token (best-first, reusing the prompt cache)
so that multi-token answers are compared as wholes.

`complete` reads a named set of multi-token outcomes **exactly**: each
outcome followed by its `closer` is scored by teacher forcing, and its
probability goes into `tracked` under its own name, with `text`, the
number of `tokens`, `p` (eight decimals) and `logp`. The closer is what
makes an outcome complete: "Mystery" is scored as `Mystery"`, so it does
not also count the mass of "Mystery Thriller". `complete_mass` is the total
over the set, which is how much of what the model says at all falls in
it. `eval/expect` then judges the set as it judges tokens, so a
791-outcome target is checked against the model outcome by outcome. A
record's own `complete` takes precedence: a probe midway through a list
closes its outcomes on the join, not the quote.

Records keep their `coords`, so a grid of conditions comes out as a grid of
readings.
""",
    inputs=(
        In("conditions", "records/record",
           "Chat-shaped records: `user` (required), `system` and `prefill` "
           "(optional), an `id`, and optionally `coords`; a record may carry "
           "its own `tracked`.", many=True),
        ADAPTER,
    ),
    emits=Emits('logits/decision', collection=True, doc='One item per input record: `id`, `coords`, `entropy_bits`, `top` (the `top_k` most probable tokens, each `{token, p, logp}`), `tracked` (each tracked token by name, `{token, p, logp}`; each complete outcome by name, `{text, tokens, p, logp}`), `complete_mass` with `complete`, and `rollout` when one was requested. The header carries `top_k`.'),
    params=(
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"yes\": \" Yes\"}` — the candidate "
          "answers of a forced choice, each recorded under `tracked`. A "
          "record's own `tracked` takes precedence.",
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
          None, fields=(
              P("top_k", "int", "How many complete outcomes to keep.", 10),
              P("max_tokens", "int", "The longest outcome, in tokens.", 8),
              P("max_forwards", "int", "The budget of model calls, the prefill included.", 128),
              P("floor", "float", "Prune a path whose whole probability falls below this.", 0.001),
              P("terminators", "list[string]",
                "An outcome is complete at the first token containing one of these.", ['"']),
          )),
        P("complete", "object",
          "Score a set of complete outcomes exactly: `{\"items\": [...], "
          "\"closer\": \"\\\"\"}`, recorded under `tracked`. A record's own "
          "`complete` takes precedence.",
          None, fields=(
              P("items", "list[string] | object",
                "The outcomes: a list, or a target spec (`weights` or `uniform`, with any "
                "`transform`) whose support is read, so a `top_k` reads a rung's own vocabulary.",
                None, fields=(TARGET_UNIFORM, TARGET_WEIGHTS, TARGET_TRANSFORM)),
              P("closer", "string", "The text that ends an outcome.", '"'),
          )),
    ),
    example={
        "model": "$model",
        "tracked": {"red": "red", "blue": "blue", "green": "green"},
    },
    example_inputs={"conditions": {"$fetch": "$conditions"}},
)

SCORE = Op(
    name="text/score",
    requires="mlx-local",
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
    requires="mlx-local",
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

CAPTURE_WEIGHTS = Op(
    name="weights/capture",
    requires="mlx-local",
    summary=(
        "Read the model's own learned tensors — their shape, norm, "
        "sparsity and, on request, their spectrum and their values — with "
        "no prompt and no forward pass."
    ),
    description="""\
Every other readout runs the model. This one reads it: what the model IS,
rather than what it did on an input, so there is nothing for the reading
to be representative of and nothing to sample.

`points` names parameters the way the module tree does —
`layers.12.self_attn.q_proj`, `embed_tokens`,
`layers.*.mlp.down_proj`, `layers.*.post_attention_layernorm` — with
`*` standing for one segment. A point that names a module takes its
parameters; `"all"` takes every tensor the model has, which on a 4B
model is several hundred.

Each item carries the cheap facts always: `shape`, `frobenius`, `mean`,
`std`, `max_abs` (where an outlier channel shows) and `sparsity`. Two
are asked for, because both are expensive in their own way:

* `spectrum: k` computes an SVD per tensor and records the top `k`
  singular values, σ₁ and the effective rank — about a second for a
  2048×1536 matrix, so name the points you mean.
* `values: true` carries the tensor itself, and is refused past two
  million numbers across the selection: a 4B model's embedding table is
  four hundred million, and the reduced forms above are what this block
  is for.

An adapter on the node's `adapter` port is fused before the read, so the
parameters captured are the adapted ones — the base plus what training
wrote. To read what training wrote BY ITSELF, `adapter/measure` reads
the adapter's own deltas and needs no model at all.
""",
    inputs=(ADAPTER,),
    emits=Emits('weights/parameter', collection=True, doc="One item per parameter tensor, id and `coords.module` naming it in the model's own tree: `shape`, `n`, `dtype`, `frobenius`, `mean`, `std`, `max_abs`, `sparsity`, plus `singular_values`/`spectral`/`effective_rank` under `spectrum` and `values` under `values`. The header carries the `model` and `captured` — the tensors read, the numbers that is, and how many the model has."),
    params=(
        P("points", "list[string] | \"all\"",
          "Which parameters to read, named as the module tree names them; "
          "`*` stands for one segment.",
          "all"),
        P("spectrum", "int",
          "Record this many singular values per tensor, with σ₁ and the "
          "effective rank. An SVD per tensor: name your points.",
          0),
        P("values", "bool",
          "Carry each tensor itself, flattened. Refused past two million "
          "numbers across the selection.",
          False),
    ),
    example={"model": "$model", "points": ["layers.*.mlp.down_proj"],
             "spectrum": 8},
)

DECOMPOSE_WEIGHTS = Op(
    name="weights/decompose",
    requires="mlx-local",
    summary=(
        "A parameter's principal directions in the residual stream — what "
        "a projection reads, or what it writes — as directions the rest of "
        "the direction family can take."
    ),
    description="""\
A weight matrix has two spaces, its input's and its output's, and for the
attention and MLP projections exactly one of them is **the residual
stream** — the space the rest of the model, and every readout, talks
about. `q_proj`, `k_proj`, `v_proj`, `gate_proj` and `up_proj` READ the
residual stream, so their right singular vectors are the directions they
ask about; `o_proj` and `down_proj` WRITE it, so their left singular
vectors are the directions they contribute. The other side is head or
hidden space, where a direction means nothing to an unembedding or to
another layer, and this block refuses it by name.

What comes out is `direction/vector` — the same kind `direction/fit`
emits from activations — so everything that family does applies:
`direction/unembed` names the tokens a direction promotes,
`direction/project` measures activations against it,
`geometry/compare` measures two of them against each other. A weight's
direction and an activation's direction are comparable when they share a
space, which is what makes this the bridge between the two halves.

**What a principal direction is not.** σ₁ of a full weight matrix is the
dominant direction of the whole map — everything that module does to
everything it sees — and it is rarely a feature. Read through the
unembedding it usually names tokens in no order a person recognises.
The directions worth reading that way come from a matrix that was
trained to do ONE thing: an adapter's delta (`adapter/measure` with
`vectors`), or a difference between two checkpoints. Use this block for
what a module's geometry IS — rank, spread, how much of the map is one
direction — and for comparing modules across layers or models.

`points` is required: an SVD per tensor is not something to do to a
whole model by accident.
""",
    inputs=(ADAPTER,),
    emits=Emits('direction/vector', collection=True, doc="`top_k` items per module, ids `<parameter>#<k>`: the unit direction, `norm` its singular value, `space` naming the model, the layer and the point the module reads or writes (`attn.in_norm`, `attn_out`, `mlp.in_norm`, `mlp_out`), and `derivation` recording the module, the side and the index. The header's `decomposed` lists what was skipped and why."),
    params=(
        P("points", "list[string]",
          "Which parameters to decompose, named as the module tree names "
          "them; `*` stands for one segment. Required — an SVD per tensor "
          "is not something to do to a whole model by accident."),
        P("top_k", "int",
          "How many directions per module, largest singular value first.",
          4),
        P("side", "\"auto\" | \"in\" | \"out\"",
          "Which side to read. `auto` takes whichever side is the residual "
          "stream; naming one takes only the modules whose residual side is "
          "that, and says why it skipped the rest.",
          "auto"),
    ),
    example={"model": "$model", "points": ["layers.*.self_attn.o_proj"],
             "top_k": 2},
)

OPS: tuple[Op, ...] = (
    INTERVENE, ABLATE_LAYERS, ABLATE_HEADS, ATTENTION_PATTERNS,
    ATTRIBUTION_LOGITS, PATCH_TRACE, RESIDUALS_DIVERGENCE, RESIDUALS_VECTORS,
    LENS_POSITIONS, LENS_TRAJECTORY, STEER_INJECT,
    GENERATE, DECISION_READ, SCORE, TOKENIZE_STATS,
    CAPTURE_WEIGHTS, DECOMPOSE_WEIGHTS,
)
