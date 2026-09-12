"""What params a block accepts — so asking for something it cannot do
fails loudly (task 000438).

Six variety jobs once declared `center: true`, all six succeeded, and
all six were uncentered: the executing runner's block predated the
parameter and simply never read it. A protocol asked for a
measurement, the platform said done, and returned a different
measurement.

**Declaration is opt-in, on purpose.** A block absent from
`ACCEPTED` is unchecked, exactly as before. Listing all forty at once
would be a refactor with no failing test behind it; listing the ones
that actually grow parameters costs nothing and closes the hole where
it was found. Add a block here when you add a param to it.

The forward-compatibility argument cuts toward strictness. A NEW
protocol running against an OLD block is precisely the case that bit
us, and "this runner's `vectors/mst` does not accept `center`" is
strictly better than a quiet wrong number.

**An incomplete declaration is its own bug** — a false refusal of a
param the block does read. `tests/test_block_params.py` reads each
declared block's module and asserts every `params.get(...)` it
contains is listed, so the table cannot drift behind the code.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Accepted by every block: the wiring, not the operation.
COMMON = frozenset({
    "records", "documents", "vectors", "matrix", "collection", "items",
    "name", "description", "dry_run", "model", "adapter", "seed",
})

#: block ref -> the params it reads, beyond COMMON.
ACCEPTED: dict[str, frozenset[str]] = {
    "~canonical/ops/chat/1": frozenset({
        "n", "start", "temperature", "top_p", "max_tokens", "system",
        "tools", "max_tool_rounds", "on_tool_error", "concurrency",
        "budget_usd", "provider_options", "base_url", "keep_fields",
        "record_requests", "limit_scope", "cassette", "cassette_mode",
        "cache", "json_mode", "logprobs", "messages", "messages_field",
        "stop", "system_field", "tool_choice", "user_field", "sandbox",
    }),
    "~canonical/ops/vectors/mst/1": frozenset({
        "bridge_sigma", "keep_edges", "center", "similarity",
    }),
    "~canonical/ops/residuals/vectors/1": frozenset({
        "layers", "position", "point", "source", "template", "label_coord",
        "skip_empty", "pool", "pool_k", "pool_skip",
    }),
    # Trajectories (task 000368).
    "~canonical/ops/trajectory/capture/1": frozenset({
        "axis", "layers", "layer", "point", "position", "positions",
        "template", "replay", "label_coord", "label_field", "vocab_top",
        "max_steps",
    }),
    "~canonical/ops/trajectory/project/1": frozenset({
        "trajectory", "direction", "keep_vectors",
    }),
    "~canonical/ops/trajectory/compare/1": frozenset({
        "a", "b", "pair_by", "threshold",
    }),
    "~canonical/ops/trajectory/aggregate/1": frozenset({
        "trajectory", "by", "as", "steps",
    }),
    # Tokenizer diagnostics (task 000377).
    "~canonical/ops/tokenize/stats/1": frozenset({
        "prefix", "vocabulary", "field", "expect_depth", "keep_items",
        "top_fragmented",
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
