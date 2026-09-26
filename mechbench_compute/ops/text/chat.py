from __future__ import annotations

from mechbench_compute.lexicon._base import In, Op, Output, P

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
A remote reply can come back empty — the allowance spent on reasoning, the
content filtered — and `on_empty` decides whether such an item is kept
marked, left out, or fails the node; it is paid for and counted either way.
A `cache` keeps a memo of remote calls by request hash, so re-running an
unchanged node costs nothing; a `cassette` replays recorded responses
without contacting the provider at all.

**Reasoning is never text.** What a model reasons before it answers —
a provider's thinking blocks, thought summaries or reasoning field, or a
local model's thinking channel — goes to the item's `reasoning`, never
into `text`, which is the reply alone; a reader of `text` (a judge, a
measure, the next turn) never sees it. The provider's payload for it —
a signature, an encrypted block — is kept verbatim, and within a tool
loop every turn goes back to the provider with its reasoning in the
provider's own form. A reply that is only reasoning is empty, with
cause `reasoning`, and keeps what it reasoned.

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
        In("intervention", "intervene/spec",
           "An intervention declared as an object — its `items` are spec "
           "items in the grammar `intervene/apply` documents — applied "
           "during every forward pass this node runs. A `direction` or "
           "`source` an item needs arrives on the port of that name. "
           "Where the node also has an inline `spec` or `intervention` "
           "param, the param wins when both are given.",
           required=False),
        In("direction", "direction/vector",
           "A direction that fills any spec item without one.", required=False),
        In("source", "activations/vector | intervene/readout",
           "A collection of `activations/vector` — a capture, intervened or "
           "not — that fills any `mean`/`resample`/`patch` item without one. "
           "A capture readout stored before 0.110.0 is read too.",
           many=True, required=False),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, an `{\"$hf_adapter\": {\"repo\": …}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('text/document', collection=True, doc="`n` items per record, ids `<record id>-s<k>`: `text` (the reply's prose, never its reasoning), `reasoning` when the model reasoned (a list of `{text, redacted?, provider, model, native?}` in the order written: `text` the readable reasoning, empty when the provider withheld it; `redacted` when there is no readable text; `native` the provider's own block, signature or encrypted payload, verbatim, which is what lets the same model be handed the turn back), `metadata.turn` (the order of the reply's parts, and any signature a text part carried) when it did, `coords` (the record's plus `sample`), `metadata.sampling` (with `ended`: how the reply ended, the same word local or remote — `end` the model ended its turn, `stop` at one of the `stop` strings, `max_tokens` cut off at the token limit, `tool_call` a tool call left unanswered when the tool rounds ran out, `filtered` ended by the provider's filter or a refusal with some prose left, `empty` no prose and no tool call, with `metadata.empty` saying why, `other` a provider reason none of these names; a remote provider's own word stays in `metadata.call.stop_reason`, and OpenAI, xAI, DeepSeek and Gemini report a stop string as a natural stop, so there it reads `end`), `metadata.call` (provider, model version, usage, cost, latency — remote only), tool runs and sandbox calls when any, and any `keep_fields` copied from the record. A remote reply with no prose and no tool call carries `metadata.empty` (`{cause, message}`). The header carries `fidelity`, `ended` (the items counted by ending, every ending present and zero when none, so a glance says whether any reply was cut off; a header without it was stored before the count existed, and its items have no `ended` either: unknown, not natural), `spend` (calls, cost, cache hits), for a remote model `empty` (`{count, by_cause, ids, policy}`, present with a count of 0 when every reply had content), and, when tools were declared, `tools` (the dialect, how many responses called one, every error with its cause)."),
    params=(
        P("budget_usd", "float",
          "The most this node may spend on provider calls, in US dollars. "
          "Required when the model is a hosted endpoint; the node stops "
          "with what it has when the cap is reached. A job-level cap, if "
          "one is set, bounds it further.",
          None),
        P("spec", "list[object]",
          "An intervention's items, applied at every forward pass of a local "
          "model — see `text/generate`. Or an `intervene/spec` arrives on the "
          "`intervention` port. Refused for a remote model.",
          None, fields=(
              P("point", "string", "Where in the forward pass to act.", "resid_post", value="point"),
              P("parameter", "string",
                "Edit this weight instead of an activation, named as the module tree names it; "
                "`*` stands for one segment.",
                None),
              P("layers", "int | list[int] | \"all\"", "Which layers the item applies to.", "all"),
              P("positions", "selector", "Which token positions.", "last"),
              P("heads", "int | list[int]", "Only these attention heads, at a point with a head axis.", None),
              P("neurons", "int | list[int]", "Only these indices along the feature axis.", None),
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
          )),
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
        P("effort", "string",
          "How hard a provider's model works on each reply: `low`, `medium`, "
          "`high`, `xhigh` or `max` on Anthropic models that have it, "
          "`minimal` to `high` on OpenAI's reasoning models. Unset leaves the "
          "model's default. Refused by name where the model has no such "
          "level, and on local weights.", None),
        P("reasoning_display", "string",
          "What an Anthropic model returns of its reasoning: `summarized` "
          "(summaries, kept in each item's `reasoning`) or `updates` (only "
          "the short progress notes it writes between tool calls). Unset "
          "returns none of it. Changes what is kept, not what the model "
          "writes.", None),
        P("prompt_cache", "string",
          "Ask Anthropic to cache each request's prompt, `5m` or `1h`, so "
          "records that share a long system prompt or conversation read it "
          "at the cache price. The spend counts cache reads and writes. "
          "Other providers cache long prompts on their own, and refuse "
          "this.", None),
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
        P("on_empty", "string",
          "Remote path only: what to do with a reply that carries no prose "
          "and no tool call. The provider says why, as a cause: "
          "`reasoning` (the output allowance went to reasoning), `filtered` "
          "(a safety filter, a refusal or a blocked prompt), `unmapped` "
          "(content in a form the adapter cannot read) or `no_content`. "
          "Every such reply is paid for, charged to the budget and "
          "checkpointed whatever this says, so a resumed run never buys it "
          "twice. `\"keep\"`: the item is emitted with empty `text` and "
          "`metadata.empty` = `{cause, message}`. `\"skip\"`: it is left out "
          "of the collection. `\"error\"`: the node fails with the "
          "provider's message — for a protocol that treats one empty reply "
          "as a broken run. The header's `empty` counts them under every "
          "choice. A local model that ends its reply at once has written "
          "an empty reply; that is model output, not a provider condition, "
          "and this does not apply to it.",
          "keep", choices=("keep", "skip", "error")),
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
        P("api", "string",
          "Which of the provider's APIs answers, for OpenAI and xAI. "
          "`\"responses\"`: the Responses API, which returns the model's "
          "reasoning as encrypted items that go back to the same model on "
          "later turns and through the tool loop; it has no `stop`, `seed` "
          "or `logprobs`. `\"chat_completions\"`: Chat Completions, where "
          "that reasoning is a token count. Unset, a model uses Chat "
          "Completions unless it needs the Responses API (GPT-6 Astra, "
          "whose tool calls Chat Completions does not support). Refused "
          "by any other provider.",
          None, choices=("chat_completions", "responses")),
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
    from mechbench_compute import chat as chat_mod
    from mechbench_compute import model_ref as model_ref_mod

    ref = params.get("model")
    if not hasattr(ref, "base_kind"):
        ref = model_ref_mod.parse(ref)
    records = inputs.get("records") or []
    if params.get("tools"):
        params = {**params, "_block_runner": ctx.executor._tool_block_runner(ctx.secrets)}
    if ref.is_endpoint:
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
