"""The parameters every block accepts: the wiring, not the operation.

A protocol node names its block and gives it params. A few of those
params are about how the node is connected — which model it runs on,
how to label what it produces, what it may spend — and every block accepts
them, so they are documented once here rather than on each of the
fifty-five pages.

What a node computes ON is not a param. Records, vectors, directions
and the rest arrive on the node's typed input ports — by an edge from
an upstream node, or inline under the node's `inputs` — and each op's
page lists its ports.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import P, Param

COMMON: tuple[Param, ...] = (
    P("model", "model",
      "The model the block runs on: a model reference, most often the "
      "protocol's `model` param (`{\"$param\": \"model\"}`) so one protocol "
      "can run against several models. A model reference may carry its "
      "own adapters; they are part "
      "of what the model means and are fused before anything else.",
      None),
    P("adapter_scale", "float",
      "Multiplies the strength of the adapter that arrives on the node's "
      "`adapter` port (only that one, not the model's own). `0` disables "
      "it; `1` is as trained.",
      None),
    P("adapter_skip_missing", "bool",
      "When the adapter names modules the model does not have, skip them "
      "and record which were skipped in the result, instead of refusing. "
      "Off by default because a silently partial adapter is a wrong "
      "measurement.",
      False),
    P("seed", "int",
      "The random seed for anything the block draws — generation "
      "temperature, resampled activations, data shuffles, and a training "
      "block's initial adapter weights. The same seed on the same "
      "hardware class reproduces the same result.",
      0),
    P("name", "string",
      "A label stamped on the record the block produces, so a person browsing "
      "results can tell what it is. Each block has a sensible default.",
      None),
    P("description", "string",
      "Free text stamped on the emitted record beside `name`.",
      ""),
    P("dry_run", "bool",
      "For blocks that call an external provider: build every request "
      "and report what would be sent and what it would cost, but send "
      "nothing.",
      False),
    P("require_resume", "map[string, string]",
      "A demand on an upstream node: `{\"records\": \"reproducible\"}` says "
      "the node feeding this port must resume at least that faithfully "
      "or start over rather than reuse a partial result. Read by the "
      "executor when planning a re-run, never by the block itself.",
      None, choices=("reproducible", "exchangeable", "state-restorable", "restart")),
)
