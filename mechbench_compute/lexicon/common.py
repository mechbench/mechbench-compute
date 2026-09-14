"""The parameters every block accepts: the wiring, not the operation.

A protocol node names its block and gives it params. A few of those
params are about how the node is connected — where its records come
from, which model it runs on, how to label what it emits — and every
block accepts them, so they are documented once here rather than on
each of the fifty-four pages.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import P, Param

COMMON: tuple[Param, ...] = (
    P("records", "list[record] | ref",
      "The records the block works on. Usually they arrive by an edge onto "
      "the node's `records` port from an upstream node; a literal list or a "
      "`{\"$fetch\": ref}` of a stored record set works too.",
      None),
    P("documents", "list[record] | ref",
      "A document corpus, for the blocks that read text rather than prompts. "
      "Interchangeable with `records`: a document is a record whose text is "
      "in `text`.",
      None),
    P("vectors", "record | ref",
      "A `residual_vectors` record — the rows some blocks measure or "
      "compare. Usually arrives by edge on the `vectors` port.",
      None),
    P("matrix", "record | ref",
      "A similarity matrix record, for the blocks that build on one (such "
      "as `geometry/mst`). Usually arrives by edge on the `similarity` port.",
      None),
    P("collection", "record | ref",
      "A collection object — a named bundle of items — for the blocks that "
      "score or look up inside one.",
      None),
    P("items", "list[object]",
      "Literal items for a block that would otherwise read them from "
      "`records`, where a protocol wants to inline a small list.",
      None),
    P("model", "string",
      "The model the block runs on: a model reference, most often the "
      "run binding `\"$model\"` so one protocol can run against several "
      "models. A model reference may carry its own adapters; they are part "
      "of what the model means and are fused before anything else.",
      None),
    P("adapter", "adapter | ref",
      "A LoRA adapter to fuse on top of the model for this node only — "
      "the output of a `adapter/train` node by edge, an "
      "`{\"$hf_adapter\": {\"repo\": …}}` reference, or a stored adapter. "
      "Fuses last, on top of any adapters the model reference itself "
      "carries.",
      None),
    P("adapter_scale", "float",
      "Multiplies the strength of the node-level `adapter` (only that "
      "one, not the model's own). `0` disables it; `1` is as trained.",
      None),
    P("adapter_skip_missing", "bool",
      "When the adapter names modules the model does not have, skip them "
      "and record which were skipped in the result, instead of refusing. "
      "Off by default because a silently partial adapter is a wrong "
      "measurement.",
      False),
    P("seed", "int",
      "The random seed for anything the block samples — generation "
      "temperature, resampled activations, data shuffles. The same seed "
      "on the same hardware class reproduces the same result.",
      0),
    P("name", "string",
      "A label stamped on the record the block emits, so a person browsing "
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
    P("require_resume", "object",
      "A demand on an upstream node: `{\"records\": \"items\"}` says the "
      "node feeding this port must be able to resume at the `items` level "
      "or start over rather than reuse a partial result. Read by the "
      "executor when planning a re-run, never by the block itself.",
      None),
)
