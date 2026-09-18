"""Ops that talk to something beyond the model in memory — a hosted
model endpoint, the Hugging Face hub, a benchmark harness — and the
ops that train and publish adapters.

Money. A node that calls a provider **must** declare `budget_usd`; the
platform refuses the protocol without it, and the executor refuses again
if one gets through. The cap is the most it may spend, and every call's
cost is recorded in the result so the bill is part of the measurement.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Output, In, Op, P
from mechbench_compute.lexicon.common import TARGET_TRANSFORM as _TRANSFORM
from mechbench_compute.lexicon.common import TARGET_UNIFORM as _UNIFORM
from mechbench_compute.lexicon.common import TARGET_WEIGHTS as _WEIGHTS
from mechbench_compute.lexicon.model import ADAPTER

_BUDGET = P("budget_usd", "float",
            "The most this node may spend on provider calls, in US dollars. "
            "Required when the model is a hosted endpoint; the node stops "
            "with what it has when the cap is reached. A job-level cap, if "
            "one is set, bounds it further.",
            None)

_PROVIDER_OPTIONS_DOC = (
    "Provider-native request fields this block does not model, **keyed by "
    "provider** — `{\"anthropic\": {\"thinking\": {…}}}` — passed through "
    "as given. A key that names no provider is refused.")

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

#: A model in a conversation that is not a participant: a moderator, a
#: judge, a summarizer. The same fields a participant has.
_AGENT_FIELDS = (
    P("name", "string", "What the others call it. Defaults to `participant-N`.", None),
    P("model", "model", "The model it runs on."),
    P("system", "string",
      "Its system prompt; `{name}`, `{participants}`, `{others}`, `{turn}` and "
      "the record's fields are filled in.",
      ""),
    P("tools", "list[string | object]", "Tools it may call.", [],
      choices=("calc", "bench.lookup"), fields=_TOOL_FIELDS),
    P("provider_options", "map[string, map[string, json]]", _PROVIDER_OPTIONS_DOC, None),
    P("temperature", "float", "Sampling temperature; the provider's default when unset.", None),
    P("top_p", "float", "Nucleus sampling threshold; the provider's default when unset.", None),
    P("max_tokens", "int", "The longest reply, in tokens.", 1024),
    P("budget_usd", "float", "Its own spend cap, within the node's.", None),
    P("channels", "list[string]", "The channels it speaks and listens on.", ["main"]),
    P("perspective", "string", "How it sees the others' messages, overriding the node's default.",
      None, choices=("others_as_user_attributed", "others_as_user_merged")),
)

CHAT = Op(
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
""",
    inputs=(
        In("records", "records/record",
           "The prompts, each with `messages`, or `system`/`user` fields, "
           "plus `id` and optionally `coords`.", many=True),
        In("cassette", "provider/cassette",
           "Recorded responses to play back instead of calling the provider "
           "— for tests and exact reproduction.", required=False),
        ADAPTER,
    ),
    output=Output('text/document', collection=True, doc="`n` items per record, ids `<record id>-s<k>`: `text`, `coords` (the record's plus `sample`), `metadata.sampling`, `metadata.call` (provider, model version, usage, cost, latency — remote only), tool runs and sandbox calls when any, and any `keep_fields` copied from the record. The header carries `fidelity`, `spend` (calls, cost, cache hits) and, when tools were declared, `tools` (the dialect, how many responses called one, every error with its cause)."),
    params=(
        _BUDGET,
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

CONVERSATION = Op(
    name="text/converse",
    requires="by-model",
    summary=(
        "Run a multi-party conversation between model participants — each "
        "seeing the shared transcript from its own side — under a declared "
        "turn policy, context-window policy and budget."
    ),
    description="""\
Two models talking is not a special mode of `chat`; it is a list of
**participants** plus three pieces of data.

**Participants.** Each is `{name, model, system, tools?, temperature?,
top_p?, max_tokens?, budget_usd?, channels?, perspective?}`. A system prompt
may use `{name}`, `{participants}`, `{others}` and `{turn}`. Local and
hosted models mix freely.

**The perspective map.** Every participant sees the same transcript with
its own messages as `assistant` and everyone else's as `user` — attributed
by name (`others_as_user_attributed`) or merged anonymously
(`others_as_user_merged`). Nobody is deceived; the roles are rendered per
viewer, and the transcript records who produced each message *and* the role
each side saw.

**The turn policy** (`turns`): `round_robin`; `speaker_names_next` (the
speaker hands off by naming someone); `moderator` (a moderator agent
chooses); `until_judge` (a judge agent, on its own channel, says when it is
finished); `until_stop` (a stop phrase). All bounded by `max_turns`.

**The window policy** (`window`): when the transcript outgrows `tokens`,
`truncate_oldest`, `sliding` (keep the opening turn too) or `summarize`
(a model call with its own cost).

One conversation runs per input record, or one in all when there are none;
`{field}` in an `opening` line takes the record's value.
""",
    inputs=(
        In("participants", "text/agent",
           "The agents, at least two, with unique names — see above. Usually "
           "given inline.", many=True),
        In("records", "records/record",
           "One conversation per record; its fields fill `{field}` "
           "placeholders in the opening. Without any, one conversation runs.",
           many=True, required=False),
    ),
    output=Output('text/transcript', collection=True, doc='One item per conversation: `text` (the visible turns as prose), `turns` (`{role, text}`), and `metadata.transcript` — the full transcript with `messages` (each `{index, participant, role_as_seen, text, call?, tool_calls?, channel?}`), `participants`, `stopped_because` and `spend_usd`. The header carries `fidelity` and `spend`.'),
    params=(
        _BUDGET,
        P("turns", "object",
          "`{\"policy\": \"round_robin\", \"max_turns\": 6, "
          "\"stop_phrases\": [...], \"moderator\": agent, \"judge\": "
          "agent}` — who speaks next and when it stops.",
          None, fields=(
              P("policy", "string", "Who speaks next.", "round_robin",
                choices=("round_robin", "speaker_names_next", "moderator", "until_judge", "until_stop")),
              P("max_turns", "int", "The most participant turns the conversation runs.", 6),
              P("stop_phrases", "list[string]",
                "End when the last message contains one of these, ignoring case — under any policy.", []),
              P("moderator", "object", "For `moderator`: the model that chooses who speaks.", None,
                fields=_AGENT_FIELDS),
              P("judge", "object", "For `until_judge`: the model that says when the conversation is done.",
                None, fields=_AGENT_FIELDS),
          )),
        P("perspective", "object",
          "`{\"default\": \"others_as_user_attributed\", \"overrides\": "
          "{name: perspective}}` — how each participant sees the others.",
          None, fields=(
              P("default", "string", "How every participant sees the others' messages.",
                "others_as_user_attributed",
                choices=("others_as_user_attributed", "others_as_user_merged")),
              P("overrides", "map[string, string]", "Participant name → the perspective it sees instead.",
                None, choices=("others_as_user_attributed", "others_as_user_merged")),
          )),
        P("window", "object",
          "`{\"policy\": \"none\" | \"truncate_oldest\" | \"sliding\" | "
          "\"summarize\", \"tokens\": n, \"summarizer\": agent}` — what to "
          "do when the transcript outgrows the context.",
          None, fields=(
              P("policy", "string", "What to drop or condense when the transcript outgrows `tokens`.",
                "none", choices=("none", "truncate_oldest", "sliding", "summarize")),
              P("tokens", "int",
                "The transcript's ceiling, counted in whitespace-separated words. `0` never windows.", 0),
              P("summarizer", "object", "For `summarize`: the model that condenses the dropped turns.",
                None, fields=_AGENT_FIELDS),
          )),
        P("opening", "list[string | object]",
          "Scripted lines the conversation starts with: a string (spoken by "
          "the script, unattributed) or `{\"participant\": name, \"text\": "
          "…}`. Nothing is spent on them.",
          None, fields=(
              P("participant", "string", "Who says it; `user` is the unattributed script.", "user"),
              P("text", "string", "The line; `{field}` takes the record's value.", ""),
          )),
        P("max_tool_rounds", "int",
          "How many times one participant's turn may call tools and be "
          "asked again.",
          3),
        P("id", "string",
          "The conversation's id when there are no input records.",
          "conversation"),
        P("base_url", "string",
          "Send every participant's requests to this endpoint instead of "
          "the provider's default.",
          None),
    ),
    example={
        "budget_usd": 2.0,
        "opening": ["Is a lighthouse a building or a machine?"],
        "turns": {"policy": "round_robin", "max_turns": 6},
    },
    example_inputs={
        "participants": [
            {"name": "Ada", "model": {"provider": "anthropic", "model": "claude-sonnet-5"},
             "system": "You are Ada. You are debating {others}.", "budget_usd": 1.0},
            {"name": "Ben", "model": "google/gemma-3-4b-it",
             "system": "You are Ben. Disagree politely."},
        ],
    },
)

JUDGE = Op(
    name="eval/judge",
    requires="by-model",
    summary=(
        "Have a model grade each record against a rubric — a score, a label "
        "or an A/B preference — with repeated votes, the spread between "
        "them, and the position order randomised and recorded."
    ),
    description="""\
Each record's `text` (for a pairwise scale, its `text_a` and `text_b`) is
shown to the judge — that field and nothing else, so it cannot see the
condition labels — together with the rubric and an instruction to answer in
JSON. A record that carries the text under another name goes through
`records/rename` first. Three commitments make the numbers usable:

* **Votes, not a verdict.** `n_votes` repeats the call. Numeric scores
  are averaged and their spread kept; labels and preferences take the
  majority, and `agreement` records how often the judge agreed with itself.
  A rubric that produces 0.5 there is a finding.
* **Position is randomised, recorded, and undone.** In pairwise mode the
  A/B order flips per vote from a seeded coin; each vote records the
  order it saw and the letter it answered, and the answer is mapped back
  to the record's own `text_a`/`text_b` before anything counts it. The
  summary reports how often the option shown FIRST won across every vote
  — 0.5 is the honest number, and 1.0 is a judge with no opinion about
  the writing.
* **Parsing is honest.** A vote that could not be read is recorded as
  unparsed rather than scored; a numeric answer outside the scale is
  clamped and flagged.
* **An empty subject is not judged.** A record whose judged field is
  missing or blank is refused by name, because a winner over an empty
  string looks exactly like every other winner in the column. This is
  reachable: a `records/zip` with `on_missing: "placeholder"` keeps the
  key of a branch that failed. `on_missing: "skip"` keeps those records
  as unjudged rows and grades the rest.

The judge runs through `chat`, so it inherits the budget cap, concurrency,
resumability and per-call provenance; a local model is the cheap first test.
""",
    inputs=(
        In("records", "records/record",
           "The subjects to grade, each with `text` — or `text_a` and "
           "`text_b` for a pairwise scale. A document collection is read the "
           "same way.", many=True),
    ),
    output=Output('eval/verdict', collection=True, doc='One item per subject: `id`, `coords`, the verdict (`score`/`spread`/`min`/`max`, or `label`/`counts`/`agreement`, or `winner`/`counts`/`agreement`), `rationale`, `n_votes`, `n_parsed`, every `vote`, `unparsed: true` when no vote could be read, and `unjudged: true` with the `missing` field names when there was nothing to judge. The header carries `judge` (who graded and how), `summary` (mean/median/stdev or counts, `n_unparsed`, `n_unjudged` and which, `first_shown_win_rate` for pairwise) and `spend`.'),
    params=(
        P("judge", "object",
          "Who grades: `{\"model\": …, \"system\": rubric, \"max_tokens\": "
          "512, \"temperature\": …, \"budget_usd\": …, "
          "\"provider_options\": …}`. `model` is required; `rubric`, when "
          "given, is appended to `system`. `temperature` is sent only if "
          "you name one — a judge's steadiness comes from `n_votes` and "
          "is reported as `agreement`, and some models refuse the "
          "parameter outright.", fields=(
              P("model", "model", "The judge's model."),
              P("system", "string", "The judge's system prompt; `rubric` is appended to it.", ""),
              P("max_tokens", "int", "The longest verdict, in tokens.", 512),
              P("temperature", "float", "Sampling temperature, sent only when named.", None),
              P("budget_usd", "float", "The spend cap, when the node sets none.", None),
              P("provider_options", "map[string, map[string, json]]", _PROVIDER_OPTIONS_DOC, None),
          )),
        P("rubric", "string",
          "The standard the judge applies, appended to `judge.system`. One "
          "of the two must be present — an unstated standard is not a "
          "measurement.",
          ""),
        P("scale", "object",
          "What the judge answers with: `{\"type\": \"numeric\", \"min\": 1, "
          "\"max\": 5}` (or `\"range\": [1, 5]`); `{\"type\": "
          "\"categorical\", \"labels\": [...]}`; or `{\"type\": "
          "\"pairwise\"}`.",
          {"type": "numeric", "min": 1, "max": 5}, fields=(
              P("type", "string", "What kind of answer.", "numeric",
                choices=("numeric", "categorical", "pairwise")),
              P("kind", "string", "The older spelling of `type`; read when `type` is absent.", None,
                choices=("numeric", "categorical", "pairwise")),
              P("min", "float", "For `numeric`: the lowest score. Defaults to `range[0]`, else 1.", None),
              P("max", "float", "For `numeric`: the highest score. Defaults to `range[1]`, else 5.", None),
              P("range", "list[float]", "For `numeric`: `[min, max]` in one field.", None),
              P("labels", "list[string]", "For `categorical`: the labels, at least two.", None),
          )),
        P("n_votes", "int", "How many times each subject is judged.", 1),
        _BUDGET,
        P("concurrency", "int",
          "How many judge requests are in flight at once (remote judges).",
          4),
        P("on_missing", "string",
          "A record whose judged field is missing or blank: `\"error\"` "
          "refuses it by name; `\"skip\"` keeps it as an unjudged row, "
          "naming what was absent, and grades the rest.",
          "error", choices=("error", "skip")),
    ),
    example={
        "judge": {"model": {"provider": "anthropic", "model": "claude-sonnet-5"},
                  "system": "You grade short stories for originality."},
        "rubric": "1 = a stock plot told plainly; 5 = a premise you have not seen before.",
        "scale": {"type": "numeric", "min": 1, "max": 5},
        "n_votes": 3,
        "budget_usd": 3.0,
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/stories"}}},
)

EVAL_HF_METRIC = Op(
    name="eval/score",
    summary=(
        "Score prediction and reference fields on a record stream with any "
        "metric from the Hugging Face `evaluate` hub — accuracy, exact "
        "match, F1, BLEU — rather than reimplementing it."
    ),
    description="""\
The named metric is loaded from the hub and computed over every record's
`prediction` against its `reference`. Each numeric value the metric
returns becomes one row, stamped with `variant` so that a base run and an
adapter run union into one table for `records/subtract`. The metric library's
version is recorded on the table, because metric definitions change across
releases.
""",
    inputs=(
        In("records", "records/record",
           "Records carrying a `prediction` and a `reference`.", many=True),
    ),
    output=Output('records/table', collection=False, doc='One row per value the metric returned: `metric`, `variant`, `value`, `n`.'),
    params=(
        P("metric", "string",
          "The hub metric's name: `\"accuracy\"`, `\"exact_match\"`, "
          "`\"f1\"`, `\"bleu\"`, `\"rouge\"`, …"),
        P("kwargs", "map[string, json]",
          "Extra keyword arguments for the metric's `compute` — e.g. "
          "`{\"average\": \"macro\"}` for F1.",
          None),
        P("variant", "string",
          "A label for which model produced the predictions, stamped on "
          "every row as a coordinate.",
          "base"),
    ),
    example={"metric": "exact_match", "variant": "adapter"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/answers"}}},
)

EVAL_SUITE = Op(
    name="eval/benchmark",
    requires="mlx-local",
    summary=(
        "Run standard benchmark tasks from the lm-evaluation-harness against "
        "the model — through mechbench's own model, so pinned revisions and "
        "fused adapters count."
    ),
    description="""\
Each named task is evaluated by the harness with the bound model wrapped as
its backend, so anything the platform knows how to load — a pinned
revision, a stacked adapter, a merged checkpoint — is what gets measured.
Every (task, metric) the harness reports becomes one row, stamped with
`variant`, ready for `records/union` and `records/subtract` against another run.

The harness version is recorded on the table: prompt templates change
between its releases, so the version is part of the measurement.
""",
    inputs=(ADAPTER,),
    output=(
        Output('records/table', collection=False, doc='One row per (task, metric): `task`, `metric`, `variant`, `value`, `stderr`, `n`.')
    ),
    params=(
        P("tasks", "list[string]",
          "The harness task names to run, e.g. `[\"hellaswag\", "
          "\"arc_easy\"]`."),
        P("limit", "int",
          "Evaluate only this many examples per task — for a quick look. "
          "By default every example.",
          None),
        P("num_fewshot", "int",
          "How many in-context examples to give. By default each task's "
          "own setting.",
          None),
        P("variant", "string",
          "A label for which model was measured, stamped on every row.",
          "base"),
    ),
    example={"model": {"$param": "model"}, "tasks": ["hellaswag", "arc_easy"], "limit": 200, "variant": "base"},
)

FINETUNE_LORA = Op(
    name="adapter/train",
    requires="mlx-local",
    summary=(
        "Train a LoRA adapter that shapes what the model says at a decision "
        "point toward a target distribution over outcomes — and emit the "
        "adapter as an object whose lineage is the training's methods "
        "section."
    ),
    description="""\
Training data are chat-shaped prompt records; at the point where the
assistant's turn begins (after any `prefill`), the model is trained toward
a **target distribution** over outcome strings rather than toward one
answer — soft-target cross-entropy. `target` describes that distribution:
`{"uniform": [outcomes]}`, or `{"weights": {outcome: weight}}` (raw corpus
frequencies, say), optionally reshaped by a `transform` chain — `sqrt`,
`pow`, `temper`, `temper_to_entropy`, `mix_uniform`, `top_k`, `normalize` —
so one frequency table can be trained flat, tempered or inverted by
declaration.

An outcome may be many tokens long ("Science Fiction Fantasy"). The
default items train the first token as a soft row and one second token
per outcome, which is exact for outcomes of one or two tokens. `path`
items train the **whole trie**: each step draws outcomes by their target
mass and trains a soft row at every token of each one, the `closer`
included, so every token is trained toward the distribution of what can
follow it, in proportion to the mass that reaches it. The closer is how an
outcome ends: after "Mystery" the row holds both the closing quote and
" Thriller".

With `depth` > 1 the outcome is a *sequence* of slots (a list of three
colours), each slot with its own target (`per_slot`) or all sharing one;
`join` and `closer` are the text between and after them. Sequences are
sampled fresh every step, and `replace: false` draws them without
replacement: a slot draws only from the outcomes not yet drawn, their
weights renormalized, so no outcome repeats.

A slot is one token (`unit: "token"`) or a whole outcome (`unit:
"item"`). With token slots the **naturalism gate** checks first that every
sampled sequence tokenizes to exactly one token per slot, so slot *i* is
token *i*: an adapter trained on a vocabulary the tokenizer splits would
be training on noise. With item slots each step trains `path` items
through the list: a soft row at every token, each over the outcomes still
available to that slot, so the rows themselves carry the no-repeats rule.
An outcome's first-slot tokens differ from its tokens after the join
("Science" against " Science"), and the gate checks every outcome in
both places.

`anchors` are prompts with a known correct `answer`, mixed into each batch
so the adapter learns the distribution without forgetting how to answer.

If the model reference already carries adapters, training happens on the
fused stack — the new round learns a delta on top. Long runs checkpoint
every `checkpoint_every` steps and resume from the same trajectory.

`seed` fixes the whole run: the adapter's initial weights as well as the
sampling order, so two runs of the same node on the same machine produce
a byte-identical adapter. Change the seed to see the spread a different
draw gives.
""",
    inputs=(
        In("records", "records/record",
           "The training prompts: chat-shaped records with `user`, and "
           "optionally `system` and `prefill`.", many=True),
        In("anchors", "records/record",
           "Prompt records with a known `answer`, mixed into each batch.",
           many=True, required=False),
    ),
    output=Output('adapter/lora', collection=False, doc="`data` (safetensors bytes), `format`, `base_model`, `trained_on` (the base and any prior adapters), `lora` (rank, alpha, scale, target modules, parameter count) and `train` (steps, lr, seed, batch, final loss, the target spec, depth, unit, replace, positions, counts). Wire it into a later node's `adapter` port, or `adapter/publish`."),
    params=(
        P("target", "object",
          "The target distribution — `{\"uniform\": [...]}` or "
          "`{\"weights\": {...}}`, with optional `transform` steps, "
          "`depth`, `join`, `per_slot`. See above.", fields=(
              _UNIFORM, _WEIGHTS, _TRANSFORM,
              P("depth", "int",
                "How many slots an outcome has. Above 1, each outcome is a sequence sampled fresh every step.", 1),
              P("join", "string", "For depth > 1: the text between slots.", ""),
              P("per_slot", "list[object]",
                "For depth > 1: one target per slot, as many as `depth`. Without it every slot shares this one.",
                None, fields=(_UNIFORM, _WEIGHTS, _TRANSFORM)),
              P("unit", "string",
                "For depth > 1: what a slot is — one token, or a whole outcome of any length, "
                "trained with `path` items.",
                "token", choices=("token", "item")),
              P("replace", "bool",
                "For depth > 1: whether an outcome can be drawn again in a later slot. "
                "`false` draws without replacement.",
                True),
          )),
        P("steps", "int", "Training steps.", 250),
        P("lr", "float", "Learning rate.", 1e-4),
        P("lora", "object",
          "`{\"rank\": 8, \"alpha\": 16, \"target_modules\": [\"q_proj\", "
          "\"v_proj\"]}` — the adapter's shape.",
          {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}, fields=(
              P("rank", "int", "The adapter's rank.", 8),
              P("alpha", "float", "The scaling numerator: the update is scaled by `alpha / rank`.", 16),
              P("target_modules", "list[string]", "The projections the adapter is trained on.",
                ["q_proj", "v_proj"],
                choices=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")),
          )),
        P("batch", "object",
          "Items per step by kind: depth 1 `{\"target\": 3, \"anchor\": 1, "
          "\"continuation\": 2}`, or `{\"path\": 3, \"anchor\": 1}` for the "
          "whole trie; token slots `{\"sequence\": 3, \"target\": 1, "
          "\"anchor\": 1}`; item slots `{\"path\": 3, \"anchor\": 1}`. A kind "
          "the target's shape does not build is refused.",
          None, fields=(
              P("target", "int", "Target items per step: at depth > 1, the first slot's marginal rows.", None),
              P("anchor", "int", "Anchor items per step.", None),
              P("continuation", "int", "Continuation items per step (depth 1).", None),
              P("sequence", "int", "Sampled sequences per step (depth > 1, token slots).", None),
              P("path", "int",
                "Outcomes (or, with item slots, whole lists) drawn per step and trained "
                "as soft rows at every token.", None),
          )),
        P("closer", "string",
          "The text after the outcome that closes the decision — `\" }\"` "
          "for depth 1, `'\"'` for deeper tries. It is how an outcome that "
          "begins another (\"Mystery\", \"Mystery Thriller\") ends, so item "
          "slots require one.",
          None),
        P("positions", "\"all\" | \"skip_first\" | list[int]",
          "For token slots: which slots are trained. `\"skip_first\"` leaves "
          "the first slot untrained.",
          "all"),
        P("marginal", "bool",
          "For token slots: also train the first slot's marginal distribution "
          "as its own item. Turn off with `positions: \"skip_first\"`.",
          True),
        P("naturalism", "bool | object",
          "Run the one-token-per-slot gate before training; `{\"samples\": "
          "40}` sets how many sequences it checks.",
          True, fields=(
              P("samples", "int", "How many sampled sequences the gate checks.", 40),
          )),
        P("checkpoint_every", "int",
          "Save resumable training state every this many steps.",
          50),
    ),
    example={
        "model": {"$param": "model"},
        "target": {"weights": {"$ref": {"bench": "you/lab/frequencies"}},
                   "transform": [{"op": "sqrt"}, {"op": "normalize"}]},
        "steps": 300,
        "lora": {"rank": 8, "alpha": 16},
        "seed": 7,
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}, "anchors": {"$ref": {"bench": "you/lab/anchors"}}},
)

MEASURE_ADAPTER = Op(
    name="adapter/measure",
    summary=(
        "Measure what training wrote, from the adapter itself: per layer and "
        "module, the norm, spectrum and effective rank of the delta, and its "
        "share of the adapter's mass — no model, no prompt, no forward pass."
    ),
    description="""\
An adapter IS a set of low-rank deltas, one per (layer, projection), and it
is already an object. Reading it answers a different question from any
capture: not what the model did on an input, but **what training changed**
— per module rather than per prompt, and in seconds rather than a forward
pass per condition.

Each item carries `frobenius` (how much was written there at all),
`spectral` and `singular_values` (how much of it is one direction),
`effective_rank` — exp(H(p)) over the normalised spectrum, so 1.0 means the
write is a line and `rank` means it fills the subspace it was given — and
`mass_share`, whose sum over the measured modules is 1: the "where did
training write" map.

The spectrum is **exact, not estimated**: `ΔW = scale · B · A` has at most
`rank` non-zero directions, so its singular values are those of an r×r
matrix built from the factors, and the delta itself is never formed.

With `vectors: true` each item also carries the principal left-singular
direction, unit length, in the module's output space. Two adapters
measured into one collection (`records/union`) are then compared by
`geometry/compare` with `by: "module"`: whether two training runs moved
the model the same way, answered in weight space rather than by capturing
what each does to a prompt.
""",
    inputs=(
        In("adapter", "adapter/lora",
           "The adapter to measure — from an `adapter/train` node or a stored "
           "one. Its bytes are read; the model it was trained on is not "
           "loaded."),
    ),
    output=Output('adapter/delta', collection=True, doc="One item per module, id and `coords.module` the module's own name in the model tree (`layers.12.self_attn.q_proj`) — two layers' `q_proj` are two modules, and `coords.projection` is what groups them: `frobenius`, `spectral`, `singular_values`, `effective_rank`, `mass_share`, `rank`, `shape`, and `vector`/`basis` when asked for. `coords` carry `layer`, `module`, `projection`, `container` and, when `source` is given, `adapter`. The header carries the adapter's `base_model`, `trained_on` and `lora` shape, and `measured` — the modules, layers and total norm the shares are shares of."),
    params=(
        P("layers", "list[int] | \"all\"",
          "Which layers to measure.",
          "all"),
        P("modules", "list[string] | \"all\"",
          "Which projections: `\"v_proj\"`, or `\"self_attn.v_proj\"` to "
          "disambiguate a name two containers share.",
          "all"),
        P("top_k", "int",
          "How many singular values to record per module, largest first.",
          4),
        P("vectors", "bool",
          "Also record each delta's principal direction, so two adapters can "
          "be compared at a module. One vector per module, the width of the "
          "module's output; a module training never wrote to has none, and "
          "carries no vector.",
          False),
        P("source", "string",
          "What to call this adapter, stamped on every item's "
          "`coords.adapter` — how a union of several stays readable.",
          None),
    ),
    example={"layers": [8, 12, 16], "vectors": True, "source": "zoo-cats"},
    example_inputs={"adapter": {"$ref": {"bench": "you/lab/adapter"}}},
)

HF_PUSH_ADAPTER = Op(
    name="adapter/publish",
    requires="remote",
    summary=(
        "Publish an adapter object to the Hugging Face hub as a PEFT LoRA "
        "repository, with a model card carrying its bench provenance."
    ),
    description="""\
The adapter's weights and config are written in PEFT's layout with a README
naming the base model, rank, alpha, target modules and training steps, and
uploaded as one commit. The result names that commit, so
`{"$hf_adapter": {"repo": …, "revision": …}}` fetches it straight back into
a protocol — the round trip.

Needs a Hugging Face write token in the job owner's vault. `dry_run` stages
the repository locally and reports the files and sizes without touching the
hub or needing a token.
""",
    inputs=(
        In("adapter", "adapter/lora",
           "The adapter to publish, usually from `adapter/train`."),
    ),
    output=(
        Output('adapter/push', collection=False, doc='`repo`, `private`, `files` (name and size), `lora`, `base_model`, `commit`, `url`, and `hf_adapter_ref` to fetch it back.')
    ),
    params=(
        P("repo", "string", "The destination, `\"<namespace>/<name>\"`."),
        P("private", "bool", "Create the repository private.", True),
        P("commit_message", "string", "The commit message.", "mechbench: adapter push"),
    ),
    example={"repo": "benjismith/spinner-fair-v1", "private": True},
    example_inputs={"adapter": {"$ref": {"bench": "you/lab/adapter"}}},
)

MERGE = Op(
    name="adapter/merge",
    summary=(
        "Collapse a model's adapter stack into one standalone checkpoint and "
        "publish it — to the bench or to the Hugging Face hub — so \"base "
        "plus these three adapters\" becomes a single thing anyone can load."
    ),
    description="""\
The `model` must be a structured reference carrying at least one adapter.
The merge never loads the model into memory: it rewrites the base's weight
shards one at a time with the adapters' deltas applied, so peak memory is
one shard. A manifest records every file's hash and the full stack that was
merged.

Destinations mirror the base grammar: `{"bench": {"name": "spinner-fair-v1"}}`
stores the checkpoint under the run's project, usable afterwards as
`{"base": {"bench": …}}`; `{"hf": {"repo": "user/repo", "private": true}}`
commits it to the hub, usable as `{"base": {"hf": "repo@sha"}}`. A failed
upload to the bench resumes: files already there with matching hashes are
skipped.

Nothing arrives by edge: the stack to merge is the `model` reference's.
""",
    inputs=(),
    output=Output('adapter/checkpoint', collection=False, doc='Where the checkpoint landed, its files with their hashes, and the stack that was merged.'),
    params=(
        P("to", "object",
          "Where to publish: `{\"bench\": {\"name\": …}}` (lowercase, "
          "digits, `-`, `_`) or `{\"hf\": {\"repo\": …, \"private\": …}}`. "
          "Exactly one.", fields=(
              P("bench", "object", "Publish to this platform as a stored checkpoint.", None,
                fields=(P("name", "string",
                          "The checkpoint's name: lowercase letters, digits, `-` and `_`, "
                          "starting with a letter or digit, at most 61 characters."),)),
              P("hf", "object", "Publish to a Hugging Face repository.", None,
                fields=(P("repo", "string", "The destination, `\"<namespace>/<name>\"`."),
                        P("private", "bool", "Create the repository private.", True))),
          )),
    ),
    example={"model": {"$param": "model"}, "to": {"bench": {"name": "spinner-fair-v1"}}},
)

TOOLS_CALC = Op(
    name="tools/calc",
    summary=(
        "Evaluate an arithmetic expression — numbers and operators only — as "
        "a tool a model may call."
    ),
    description="""\
The expression is parsed and refused if it contains anything but numeric
literals and `+ - * / // % **` (and parentheses): a tool a model can steer
must not be an evaluator. Offered to a `chat` or `converse` node by
naming `"calc"` in its `tools`; the model's call supplies `arguments:
{expression}`. A tool has no input ports — its arguments come from the
call.
""",
    inputs=(),
    output=None,
    params=(
        P("expression", "string",
          "The expression, when the block is run directly rather than as a "
          "tool call.",
          None),
    ),
    example={"expression": "(3 + 4) * 12 / 7"},
)

TOOLS_BENCH_LOOKUP = Op(
    name="tools/lookup",
    summary=(
        "Fetch a stored bench object by path — a tool that lets a model "
        "consult what the platform already knows."
    ),
    description="""\
Returns the object's payload, or one field of it when the call names a
`field`. Offered to a `chat` or `converse` node as `"bench.lookup"`;
the model's call supplies `arguments: {path, field?}`. Every fetch is
recorded on the item that made it. A tool has no input ports — its
arguments come from the call.
""",
    inputs=(),
    output=None,
    params=(
        P("path", "string",
          "The object path, when the block is run directly rather than as "
          "a tool call.",
          None),
        P("fetch", "callable",
          "For tests: the function used to fetch, in place of the bench "
          "client. Not for protocols.",
          None),
    ),
    example={"path": "benjismith/stories/word-lists/opening-phrases"},
)

OPS: tuple[Op, ...] = (
    CHAT, CONVERSATION, JUDGE, EVAL_HF_METRIC, EVAL_SUITE, FINETUNE_LORA,
    MEASURE_ADAPTER, HF_PUSH_ADAPTER, MERGE, TOOLS_CALC, TOOLS_BENCH_LOOKUP,
)
