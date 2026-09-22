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
  tokenized as a continuation of the RENDERED prompt, so spell it the
  way the model's next token is spelled there: after a raw prompt that
  is usually "` Paris`" with its leading space; after a chat template's
  assistant prefix it is usually "`Paris`" without one, because the
  template ends the prompt at a turn boundary. The two are different
  tokens, and the wrong one measures a token the model was never going
  to say. Every condition reports the model's own top-1 beside a target
  that differs from it, and the header counts them as `n_off_top1`:
  read that number, and each condition's `baseline_logp`, before any Δ.
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

from mechbench_compute.lexicon._base import In, Op, P

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
             "of the rendered prompt: with a leading space after a raw prompt, "
             "without one after a chat template's assistant prefix. A record's "
             "own `tracked` takes precedence; with none named, the model's own "
             "top-1 prediction for that prompt is the target, and a target "
             "that differs from it is reported beside it.",
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
      "The direction, usually a stored one (`{\"$ref\": …}`); or it arrives on the node's `direction` port.", None),
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
    P("except", "bool",
      "Invert the sets this item names — every layer, head or neuron BUT "
      "those — which measures a circuit's completeness where the direct "
      "ablation measures its faithfulness.",
      False),
    P("from", "object",
      "For `patch`: where the row is read, when that is not where it is "
      "written — the patchscope's move.", None, fields=(
          P("layer", "int", "The source layer."),
          P("point", "string", "The source point; the item's own by default.", None, value="point"),
      )),
    P("pattern", "object",
      "At `attn.scores` or `attn.weights`: the attention EDGE to act on — "
      "which source positions the selected destinations may attend to.",
      None, fields=(
          P("from", "selector", "The source (key) positions."),
          P("to", "selector", "The destination (query) positions; the item's `positions` by default.", None),
      )),
    P("renormalize", "bool",
      "At `attn.weights`, after zeroing: rescale the rows that lost mass so "
      "they sum to one again. The rows that lost none are left as they are.",
      True),
    P("seed", "int", "The seed `resample` draws with; the node's `seed` by default.", None),
    P("sweep_over", "list[string]",
      "Which of the node's sweep axes vary THIS item — `[\"layers\"]` sweeps "
      "this item's layers and leaves its strength alone. Every axis, by "
      "default.",
      None, choices=("strength", "layers", "heads", "positions", "neurons")),
    P("side", "string",
      "For a weight's `project_out`: the side facing the residual stream, where the module's "
      "name does not imply it.",
      None, choices=("in", "out")),
    P("rank", "int", "For a weight's `truncate`: how many singular directions to keep.", None),
)

#: The port every op that runs a forward pass may take: an intervention
#: declared as an object, live at every position the pass visits — the
#: activation-side twin of ADAPTER (000601).
INTERVENTION = In("intervention", "intervene/spec",
                  "An intervention declared as an object — its `items` are spec "
                  "items in the grammar `intervene/apply` documents — applied "
                  "during every forward pass this node runs. A `direction` or "
                  "`source` an item needs arrives on the port of that name. "
                  "Where the node also has an inline `spec` or `intervention` "
                  "param, the param wins when both are given.",
                  required=False)

_DIRECTION_PORT = In("direction", "direction/vector",
                     "A direction that fills any spec item without one.", required=False)

_SOURCE_PORT = In("source", "activations/vector | intervene/readout",
                  "A collection of `activations/vector` — a capture, intervened or "
                  "not — that fills any `mean`/`resample`/`patch` item without one. "
                  "A capture readout stored before 0.110.0 is read too.",
                  many=True, required=False)

#: The params an op that takes an intervention shares with `intervene/apply`.
_SWEEP_PARAMS = (
    P("sweep", "object",
      "The axes to vary, each a list of values the spec items' field of "
      "that name takes in turn: `{\"strength\": [0.5, 1.0, 2.0]}` scales "
      "every item's `strength`; `{\"layers\": [0, 1, 2]}` runs the spec at "
      "each layer; `heads`, `positions` and `neurons` likewise. Several "
      "axes are a cartesian product, run with `strength` outermost, and "
      "each becomes a coordinate on every row — `factor`, `layer`, `head`, "
      "`position`, `neuron` — so a sweep is summarised, compared and "
      "plotted on the axis it varied.",
      {"strength": [1.0]}, fields=(
          P("strength", "list[float]", "The factors; `0` is the untouched model.", [1.0]),
          P("layers", "list[json]",
            "The layers, one cell each: `[0, 1, 2]`, or `[[0,1],[2,3]]` for "
            "groups. The coordinate is the layer, or `0+1` for a group.", None),
          P("heads", "list[json]", "The attention heads, one cell each.", None),
          P("positions", "list[selector]", "The position selectors, one cell each.", None),
          P("neurons", "list[json]", "The feature indices, one cell each.", None),
      )),
    P("control", "bool",
      "Add a factor‑0 run — the model untouched — to the sweep, so every "
      "record has a baseline (`factor: 0.0`). One run however many axes "
      "the sweep has: an unintervened pass does not depend on the layer "
      "the intervention would have named. Set `false` when the sweep "
      "already contains a strength of `0` or no baseline is wanted.",
      True),
)

_PROMPTS = In("records", "records/record",
              "The prompts, one per record; a record's prompt is its `user`, "
              "`prompt` or `text` field. A record may carry its own `tracked`.",
              many=True)


# --- editing the forward pass ------------------------------------------------------


# --- reading and generating --------------------------------------------------------


OPS: tuple[Op, ...] = (
    
    
    
    
    
    
)
