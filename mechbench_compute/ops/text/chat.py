from __future__ import annotations

from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.lexicon.external import _BUDGET
from mechbench_compute.lexicon.model import (
    _DIRECTION_PORT,
    _SOURCE_PORT,
    _SPEC_FIELDS,
    _SWEEP_PARAMS,
    ADAPTER,
    INTERVENTION,
)

#: A tool a chat may call: a built-in by name, or a definition.
_TOOL_FIELDS = (
    P("name", "string", "The tool's name, unique among the tools offered."),
    P("description", "string", "What the tool does, as the model is told.", ""),
    P("schema", "json",
      "The arguments, as a JSON Schema object (`input_schema` is read too). "
      "By default a tool takes no arguments.",
      None),
    P("handler", "object",
      "What runs when the model calls it: one of `block` (an operation, with "
      "`params`), `protocol`, or `sandbox` (a sandbox method). Without one the "
      "tool is offered and a call returns an error.",
      None,
      fields=(
          P("block", "string", "The operation that runs the call.", None),
          P("params", "json", "The operation's params; the call's arguments are merged over them.", None),
          P("protocol", "string", "A protocol that runs the call.", None),
          P("sandbox", "string", "The sandbox method that runs the call.", None),
      )),
)


#: One message of a conversation sent as-is.
_MESSAGE_FIELDS = (
    P("role", "string", "Who said it. A system prompt goes in `system`, not here.",
      "user", choices=("user", "assistant")),
    P("content", "string | list[string | object]",
      "The text, or a list of parts: text, a tool call, a tool result.",
      fields=(
          P("type", "string", "Which kind of part.", "text",
            choices=("text", "tool_call", "tool_result")),
          P("text", "string", "For a text part: the text.", None),
          P("id", "string", "For a tool call: its id.", None),
          P("name", "string", "For a tool call: the tool's name.", None),
          P("arguments", "json", "For a tool call: the arguments object.", None),
          P("tool_call_id", "string", "For a tool result: the call it answers.", None),
          P("content", "json", "For a tool result: what the tool returned.", None),
          P("is_error", "bool", "For a tool result: whether the call failed.", False),
      )),
)


_SANDBOX_FIELDS = (
    P("base", "string",
      "The guest the sandbox runs: `mbshell` (a shell) or `cpython` (Python 3), "
      "or a path to a `.wasm` file.",
      "mbshell"),
    P("tools", "list[string]", "The sandbox's tools offered to the model.",
      ["bash", "read_file", "write_file", "list"],
      choices=("bash", "find", "grep", "python", "read_file", "write_file", "list")),
    P("limits", "object", "The ceilings one sandbox session runs under.", None,
      fields=(
          P("memory_mb", "int", "Memory, in megabytes.", 256),
          P("fuel", "int", "Instructions executed before the guest is stopped.", 100_000_000_000),
          P("wall_seconds", "float", "Wall-clock time per call, in seconds.", 30),
          P("output_bytes", "int", "The most output one call returns.", 262144),
          P("max_files", "int", "How many files the workspace may hold.", 10000),
          P("max_bytes", "int", "How many bytes the workspace may hold.", 268435456),
      )),
    P("strict", "bool",
      "Virtualise the clock and the random number generator, so a run is a "
      "function of its inputs.",
      False),
    P("snapshot", "json",
      "The workspace a session starts from: a path → content map, or a stored "
      "`sandbox/snapshot`.",
      None),
    P("mounts", "list[object]", "Further trees mounted into the workspace.", None,
      fields=(
          P("path", "string", "Where the tree is mounted (`at` is read too)."),
          P("snapshot", "json", "The tree itself: a path → content map.", None),
          P("object", "string", "A stored object to mount, by reference.", None),
          P("digest", "string", "The stored object's expected digest.", ""),
      )),
)


OP = Op(
    name="text/chat",
    requires="by-model",
    summary=(
        "Send each record's prompt to a model as a chat — local weights or a "
        "hosted endpoint, the same node either way — and collect the "
        "replies, with tool use, cost and provenance recorded per item."
    ),
    description="""\
Which side of the network answers is decided by the `model`:
`"google/gemma-3-4b-it"` runs on this machine; `{"provider": "anthropic",
"model": "claude-…"}` goes to that provider's API. Everything downstream
reads the same document collection either way, so a protocol comparing a
local fine-tune against a frontier model is one graph with two chat nodes.

Each record supplies either a full `messages` conversation or the
`system`/`user` fields a corpus record has (the same fields `generate`
reads, so a local generate node can be swapped for a chat node without
rewriting the corpus). `n` replies are sampled per record, indices
`start` … `start + n − 1`.

**Tools.** `tools` offers the model functions to call — the built-ins by
name (`"calc"`, `"bench.lookup"`) or full definitions `{name, description,
schema, handler: {"block": …}}` whose handler is any op. A reply that calls
a tool is answered and the model asked again, up to `max_tool_rounds`
times; every tool run is recorded on the item. A `sandbox` adds a
per-item filesystem sandbox with its own tools and records its calls and
final snapshot.

**What a run can promise.** Local sampling is reproducible: seeds make each
item a pure function of its key. A remote call is exchangeable at best —
someone else's sampler, possibly a different model version tomorrow — so
each item records the model version that answered, the cost and the usage.
A `cache` keeps a memo of remote calls by request hash, so re-running an
unchanged node costs nothing; a `cassette` replays recorded responses
without contacting the provider at all.

**An intervention** — inline `spec` items, or an `intervene/spec` on the
`intervention` port —
is live at every forward pass a LOCAL model runs for this node, prefill
and every decoding step alike, exactly as `text/generate` documents; a
`sweep` gives one set of replies per factor. A remote model has no
forward pass to intervene on, so an intervention on one is refused by
name.
""",
    inputs=(
        In("records", "records/record",
           "The prompts, each with `messages`, or `system`/`user` fields, "
           "plus `id` and optionally `coords`.", many=True),
        In("cassette", "provider/cassette",
           "Recorded responses to play back instead of calling the provider "
           "— for tests and exact reproduction.", required=False),
        INTERVENTION,
        _DIRECTION_PORT,
        _SOURCE_PORT,
        ADAPTER,
    ),
    output=Output('text/document', collection=True, doc="`n` items per record, ids `<record id>-s<k>`: `text`, `coords` (the record's plus `sample`), `metadata.sampling`, `metadata.call` (provider, model version, usage, cost, latency — remote only), tool runs and sandbox calls when any, and any `keep_fields` copied from the record. The header carries `fidelity`, `spend` (calls, cost, cache hits) and, when tools were declared, `tools` (the dialect, how many responses called one, every error with its cause)."),
    params=(
        _BUDGET,
        P("spec", "list[object]",
          "An intervention's items, applied at every forward pass of a local "
          "model — see `text/generate`. Or an `intervene/spec` arrives on the "
          "`intervention` port. Refused for a remote model.",
          None, fields=_SPEC_FIELDS),
        *_SWEEP_PARAMS,
        P("messages", "string | list[string | object]",
          "A conversation to send when a record has neither `messages` nor "
          "a `user` field.",
          None, fields=_MESSAGE_FIELDS),
        P("system", "string",
          "A system prompt used when the record has none.",
          None),
        P("n", "int", "Replies to sample per record.", 1),
        P("start", "int",
          "The first sample index; later nodes with a later `start` extend "
          "the collection without re-sampling.",
          0),
        P("temperature", "float",
          "Sampling temperature. Local default 0.9; remote default is the "
          "provider's.",
          None),
        P("top_p", "float",
          "Nucleus sampling threshold. Local default 0.95; remote default is "
          "the provider's.",
          None),
        P("max_tokens", "int", "The longest reply, in tokens.", 1024),
        P("stop", "list[string]",
          "Strings at which generation stops; the marker itself is not part "
          "of the reply. Honoured on both paths — the local sampler ends the "
          "sample at the first of them.",
          None),
        P("json_mode", "bool",
          "Ask the provider for a JSON-only reply. Remote only, and refused "
          "by name on a provider that cannot do it — never silently dropped.",
          False),
        P("logprobs", "int | bool",
          "Ask the provider to return token log-probabilities: an int is the "
          "top-k. Remote only, and refused by name where the provider has no "
          "logprobs or a lower ceiling. For a local model, `logits/read` "
          "reads the distribution itself.",
          None),
        P("tools", "list[string | object]",
          "Tools the model may call: built-in names, or full definitions "
          "with an op as handler.",
          None, choices=("calc", "bench.lookup"), fields=_TOOL_FIELDS),
        P("tool_choice", "string | map[string, json]",
          "How the provider should choose tools — `\"auto\"`, `\"none\"`, "
          "or a specific tool — in the provider's own vocabulary. Remote "
          "only, and refused with no `tools` declared: locally the model's "
          "chat template offers the tools and does not constrain the choice.",
          None),
        P("max_tool_rounds", "int",
          "How many times a reply may call tools and be asked again.",
          3),
        P("on_tool_error", "string",
          "Local path only. `\"record\"`: a failed tool call is recorded on "
          "the item and the run continues. `\"fail\"`: it fails the node.",
          "record", choices=("record", "fail")),
        P("sandbox", "object",
          "Give each item a filesystem sandbox: `{base, tools, limits, "
          "strict, snapshot, mounts}`, or `{}` for the default image. Its "
          "tools are offered alongside `tools`.",
          None, fields=_SANDBOX_FIELDS),
        P("provider_options", "map[string, map[string, json]]",
          "Provider-native request fields this block does not model, **keyed "
          "by provider** — `{\"anthropic\": {\"thinking\": {…}}}` — merged "
          "over any the model reference carries and passed through as given. "
          "A key that names no provider is refused: written at the top level "
          "it would reach nobody.",
          None),
        P("base_url", "string",
          "Send requests to this endpoint instead of the provider's default "
          "— a proxy or a compatible self-hosted server.",
          None),
        P("concurrency", "int",
          "How many remote requests are in flight at once.",
          4),
        P("limit_scope", "string",
          "The rate-limit bucket to share. Defaults to a fingerprint of the "
          "credential, since limits are per account. This block's own "
          "limiter, not the provider's — nothing about it goes on the wire.",
          None),
        P("cache", "string | bool",
          "Keep a memo of remote calls keyed by request hash, so unchanged "
          "requests are not paid for twice. A stored label "
          "(`\"<owner>/<project>/memos/<name>\"`), or `true` to derive one "
          "from the protocol and node ids.",
          None),
        P("cassette_mode", "string",
          "`\"replay\"`: only recorded responses, refuse anything else. "
          "`\"record\"`: call the provider and record. `\"auto\"`: replay "
          "what is recorded, call and record the rest.",
          "replay", choices=("replay", "record", "auto")),
        P("record_requests", "bool",
          "Also store each outgoing request body on its call record, not "
          "only the response. About what is kept, not what is sent: this "
          "block writes the record either way.",
          False),
        P("keep_fields", "list[string]",
          "Record fields to copy onto each output item — a reference "
          "answer, a condition label the readout needs.",
          None),
    ),
    example={
        "model": {"provider": "anthropic", "model": "claude-sonnet-5"},
        "budget_usd": 5.0,
        "n": 2,
        "max_tokens": 400,
        "tools": ["calc"],
        "cache": True,
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    """text/chat (task 000337): one block for local
    weights and remote endpoints. The ModelRef decides which — an
    endpoint ref goes to the provider transport, anything else to
    MLX through the usual model-block path, so a chat node with a
    LoRA adapter still fuses its stack."""
    from mechbench_compute import chat as chat_mod
    from mechbench_compute import model_ref as model_ref_mod

    ref = params.get("model")
    if not hasattr(ref, "base_kind"):
        ref = model_ref_mod.parse(ref)
    records = inputs.get("records") or []
    if params.get("tools"):
        # Injected AFTER the fingerprint is computed, so a callable
        # never reaches a node's identity or its emitted params.
        params = {**params, "_block_runner": ctx.executor._tool_block_runner(ctx.secrets)}
    if ref.is_endpoint:
        # A remote model has no forward pass to intervene on (000601).
        if params.get("spec") or inputs.get("intervention") is not None:
            raise ValueError(
                "text/chat: an intervention needs local weights — a remote "
                "model has no forward pass to act on. Drop the intervention, "
                "or give the node a local model reference.")
        memo = ctx.executor._open_memo(params)
        out = chat_mod.run_remote(
            ref, records, params, secrets=ctx.secrets,
            cassette=(memo.tape if memo else
                      inputs.get("cassette")),
            cassette_mode="auto" if memo else None,
            limiter=ctx.executor._limiter, job_budget=ctx.executor._budget,
            on_item=ctx.on_item, on_start=ctx.on_start,
            resume_items=ctx.resume_items)
        if memo:
            out = ctx.executor._close_memo(memo, out)
        return out
    return ctx.executor._run_model_block(
        ctx.executor._block_chat_local, inputs, {**params, "model": ref},
        on_item=ctx.on_item, on_start=ctx.on_start, resume_items=ctx.resume_items)
