"""What params a block accepts — so asking for something it cannot do
fails loudly (task 000438), for every op (task 000478).

Six variety jobs once declared `center: true`, all six succeeded, and
all six were uncentered: the executing runner's block predated the
parameter and simply never read it. A protocol asked for a
measurement, the platform said done, and returned a different
measurement.

**Every registered op is declared.** 000438 made declaration opt-in —
"listing all forty at once would be a refactor with no failing test
behind it" — and that left 46 of 54 ops unchecked, which is the same
hole standing open everywhere it had not yet been found. 000478 is that
refactor; `tests/test_block_params.py` is the test behind it, and it
fails when a new op arrives undeclared.

The forward-compatibility argument cuts toward strictness. A NEW
protocol running against an OLD block is precisely the case that bit
us, and "this runner's `vectors/mst` does not accept `center`" is
strictly better than a quiet wrong number.

**Both directions are bugs.** An incomplete declaration falsely refuses
a param the block does read. An over-declaration accepts one it never
reads — which is the original failure wearing a different hat, and
`vectors/mst` was doing it: `similarity` is an input PORT, read from
`inputs`, and declaring it as a param meant a protocol could set it and
be ignored. The test asserts the table EQUALS what the code reads,
following `params` across modules and into packages to find out.

A block absent from `ACCEPTED` is still unchecked at runtime — an
extension's op is nobody's business but its own (000410) — but no
canonical op may be absent, and the test enforces that.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Accepted by every block: the wiring, not the operation.
COMMON = frozenset({
    "records", "documents", "vectors", "matrix", "collection", "items",
    "name", "description", "dry_run", "model", "adapter", "seed",
    # Adapter fusion wiring, read by the executor wrapper, not the block.
    "adapter_scale", "adapter_skip_missing",
    # A consumer's demand on its upstream's resume level (epic 000320):
    # authored in params, read by the EXECUTOR, never by a block. It is
    # not underscore-prefixed because a protocol writes it.
    "require_resume",
})

#: block ref -> the params it reads, beyond COMMON.
ACCEPTED: dict[str, frozenset[str]] = {
    "~canonical/ops/ablate/heads/1": frozenset({
        "layers", "target", "template",
    }),
    "~canonical/ops/ablate/layers/1": frozenset({
        "component", "layers", "target", "template",
    }),
    "~canonical/ops/attention/patterns/1": frozenset({
        "layers", "template",
    }),
    "~canonical/ops/attribution/logits/1": frozenset({
        "apply_ln", "layers", "per_head_layers", "target", "template",
    }),
    "~canonical/ops/chat/1": frozenset({
        "base_url", "budget_usd", "cache", "cassette", "cassette_mode",
        "concurrency", "json_mode", "keep_fields", "limit_scope",
        "logprobs", "max_tokens", "max_tool_rounds", "messages",
        "messages_field", "n", "on_tool_error", "provider_options",
        "record_requests", "sandbox", "start", "stop", "system",
        "system_field", "temperature", "tool_choice", "tools", "top_p",
        "user_field",
    }),
    "~canonical/ops/conversation/1": frozenset({
        "base_url", "budget_usd", "id", "max_tool_rounds", "opening",
        "participants", "perspective", "turns", "window",
    }),
    "~canonical/ops/decision-read/1": frozenset({
        "conditions", "outcomes", "prefill_field", "rollout",
        "system_field", "user_field",
    }),
    "~canonical/ops/direction/add/1": frozenset({
        "directions", "weights",
    }),
    "~canonical/ops/direction/average/1": frozenset({
        "directions",
    }),
    "~canonical/ops/direction/from-pca/1": frozenset({
        "component", "label", "layer", "point", "source",
    }),
    "~canonical/ops/direction/from-vectors/1": frozenset({
        "layer", "negative", "point", "positive", "source",
    }),
    "~canonical/ops/direction/normalize/1": frozenset({
        "direction",
    }),
    "~canonical/ops/direction/orthogonalize/1": frozenset({
        "against", "direction",
    }),
    "~canonical/ops/direction/project/1": frozenset({
        "direction",
    }),
    "~canonical/ops/direction/similarity/1": frozenset({
        "a", "b", "directions",
    }),
    "~canonical/ops/direction/vocab/1": frozenset({
        "direction", "top_k",
    }),
    "~canonical/ops/eval/expectation/1": frozenset({
        "expectations", "results",
    }),
    "~canonical/ops/eval/hf-metric/1": frozenset({
        "kwargs", "metric", "prediction_field", "reference_field",
        "variant",
    }),
    "~canonical/ops/eval/suite/1": frozenset({
        "limit", "num_fewshot", "tasks", "variant",
    }),
    "~canonical/ops/factor-cross/1": frozenset({
        "axes", "factors",
    }),
    "~canonical/ops/finetune/lora/1": frozenset({
        "anchors", "answer_field", "batch", "checkpoint_every", "closer",
        "lora", "lr", "marginal", "naturalism", "positions",
        "prefill_field", "steps", "system_field", "target", "user_field",
    }),
    "~canonical/ops/generate/1": frozenset({
        "fidelity", "max_tokens", "n", "start", "system_field",
        "temperature", "top_p", "user_field",
    }),
    "~canonical/ops/grid/1": frozenset({
        "axes", "factors",
    }),
    "~canonical/ops/group-stats/1": frozenset({
        "by", "on_missing", "value",
    }),
    "~canonical/ops/hf/push-adapter/1": frozenset({
        "commit_message", "private", "repo",
    }),
    "~canonical/ops/intervene/1": frozenset({
        "control", "outcomes", "readout", "spec", "sweep", "template",
        "top_k", "track",
    }),
    "~canonical/ops/judge/1": frozenset({
        "budget_usd", "concurrency", "fields", "judge", "n_votes",
        "pairwise_fields", "rubric", "scale",
    }),
    "~canonical/ops/lens-trajectory/1": frozenset({
        "prefill_field", "system_field", "user_field",
    }),
    "~canonical/ops/lens/positions/1": frozenset({
        "layers", "target", "template",
    }),
    "~canonical/ops/merge/1": frozenset({
        "to",
    }),
    "~canonical/ops/paired-delta/1": frozenset({
        "baseline_where", "match_on", "value",
    }),
    "~canonical/ops/patch/trace/1": frozenset({
        "layers", "metric", "point", "target", "template",
    }),
    "~canonical/ops/reduce/histogram/1": frozenset({
        "bins", "hi", "lo", "value",
    }),
    "~canonical/ops/reduce/sum/1": frozenset({
        "value",
    }),
    "~canonical/ops/reduce/top-k/1": frozenset({
        "k", "value",
    }),
    "~canonical/ops/residuals/divergence/1": frozenset({
        "layers", "point", "template",
    }),
    "~canonical/ops/residuals/vectors/1": frozenset({
        "label_coord", "layers", "point", "pool", "pool_k", "pool_skip",
        "position", "skip_empty", "source", "template",
    }),
    "~canonical/ops/score/1": frozenset({
        "collection_path",
    }),
    "~canonical/ops/select/1": frozenset({
        "fields", "where",
    }),
    "~canonical/ops/steer/inject/1": frozenset({
        "alphas", "direction", "layer", "position", "template", "top_k",
        "track", "tracks",
    }),
    "~canonical/ops/table/from-records/1": frozenset({
        "row_axis",
    }),
    "~canonical/ops/template/1": frozenset({
        "templates",
    }),
    "~canonical/ops/text/stats/1": frozenset({
        "field", "keep", "measures", "mode",
    }),
    "~canonical/ops/tokenize/stats/1": frozenset({
        "expect_depth", "field", "keep_items", "prefix", "top_fragmented",
        "vocabulary",
    }),
    "~canonical/ops/tools/bench-lookup/1": frozenset({
        "fetch", "path",
    }),
    "~canonical/ops/tools/calc/1": frozenset({
        "expression",
    }),
    "~canonical/ops/trajectory/aggregate/1": frozenset({
        "as", "by", "steps", "trajectory",
    }),
    "~canonical/ops/trajectory/capture/1": frozenset({
        "axis", "label_coord", "label_field", "layer", "layers",
        "max_steps", "point", "position", "positions", "project", "reduce",
        "replay", "steps", "template", "vocab_top",
    }),
    "~canonical/ops/trajectory/compare/1": frozenset({
        "a", "b", "pair_by", "threshold",
    }),
    "~canonical/ops/trajectory/project/1": frozenset({
        "direction", "keep_vectors", "trajectory",
    }),
    "~canonical/ops/union/1": frozenset({
        "batch_axis",
    }),
    "~canonical/ops/vectors/mst/1": frozenset({
        "bridge_sigma", "center", "keep_edges",
    }),
    "~canonical/ops/vectors/similarity/1": frozenset(),
    "~canonical/ops/viz/spec/1": frozenset({
        "encoding", "mark", "title", "x", "y",
    }),
}


def check_params(block: str, params: Mapping[str, object]) -> None:
    """Raise if `block` was handed a param it does not read.

    The message names the parameter AND the block, because the useful
    question when this fires is 'does this runner's copy of that block
    know about it?' — usually the answer is that the runner is old.
    """
    accepted = ACCEPTED.get(block)
    if accepted is None:
        return
    # Underscore keys are the executor's own injections (a block
    # runner, a resume handle), never something a protocol declared.
    unknown = sorted(k for k in set(params) - accepted - COMMON
                     if not k.startswith("_"))
    if not unknown:
        return
    known = ", ".join(sorted(accepted | COMMON))
    raise ValueError(
        f"{block} does not accept {', '.join(repr(u) for u in unknown)}. "
        f"If the protocol is newer than this runner, the runner's copy of "
        f"the block predates the parameter — check its compute version. "
        f"Accepted here: {known}."
    )
