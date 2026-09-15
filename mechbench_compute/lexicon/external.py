"""Ops that talk to something beyond the model in memory — a hosted
model endpoint, the Hugging Face hub, a benchmark harness — and the
ops that train and publish adapters.

Money. A node that calls a provider **must** declare `budget_usd`; the
platform refuses the protocol without it, and the executor refuses again
if one gets through. The cap is the most it may spend, and every call's
cost is recorded in the result so the bill is part of the measurement.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import Emits, In, Op, P
from mechbench_compute.lexicon.model import ADAPTER

_BUDGET = P("budget_usd", "float",
            "The most this node may spend on provider calls, in US dollars. "
            "Required when the model is a hosted endpoint; the node stops "
            "with what it has when the cap is reached. A job-level cap, if "
            "one is set, bounds it further.",
            None)

CHAT = Op(
    name="text/chat",
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
    emits=Emits('text/document', collection=True, doc="`n` items per record, ids `<record id>-s<k>`: `text`, `coords` (the record's plus `sample`), `metadata.sampling`, `metadata.call` (provider, model version, usage, cost, latency — remote only), tool runs and sandbox calls when any, and any `keep_fields` copied from the record. The header carries `fidelity`, `spend` (calls, cost, cache hits) and, when tools were declared, `tools` (the dialect, how many responses called one, every error with its cause)."),
    params=(
        _BUDGET,
        P("messages", "list[object]",
          "A conversation to send when a record has neither `messages` nor "
          "a `user` field.",
          None),
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
          None),
        P("tool_choice", "string | object",
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
          "record"),
        P("sandbox", "object",
          "Give each item a filesystem sandbox: `{base, tools, limits, "
          "strict, snapshot, mounts}`, or `{}` for the default image. Its "
          "tools are offered alongside `tools`.",
          None),
        P("provider_options", "object",
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
          "replay"),
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
    example_inputs={"records": {"$fetch": "$prompts"}},
)

CONVERSATION = Op(
    name="text/converse",
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
    emits=Emits('text/transcript', collection=True, doc='One item per conversation: `text` (the visible turns as prose), `turns` (`{role, text}`), and `metadata.transcript` — the full transcript with `messages` (each `{index, participant, role_as_seen, text, call?, tool_calls?, channel?}`), `participants`, `stopped_because` and `spend_usd`. The header carries `fidelity` and `spend`.'),
    params=(
        _BUDGET,
        P("turns", "object",
          "`{\"policy\": \"round_robin\", \"max_turns\": 6, "
          "\"stop_phrases\": [...], \"moderator\": agent, \"judge\": "
          "agent}` — who speaks next and when it stops.",
          None),
        P("perspective", "object",
          "`{\"default\": \"others_as_user_attributed\", \"overrides\": "
          "{name: perspective}}` — how each participant sees the others.",
          None),
        P("window", "object",
          "`{\"policy\": \"none\" | \"truncate_oldest\" | \"sliding\" | "
          "\"summarize\", \"tokens\": n, \"summarizer\": agent}` — what to "
          "do when the transcript outgrows the context.",
          None),
        P("opening", "list[string | object]",
          "Scripted lines the conversation starts with: a string (spoken by "
          "the script, unattributed) or `{\"participant\": name, \"text\": "
          "…}`. Nothing is spent on them.",
          None),
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
* **Position is randomised and recorded.** In pairwise mode the A/B order
  flips per vote from a seeded coin and each vote records the order it saw;
  the summary reports how often the first-shown option won (0.5 is the
  honest number).
* **Parsing is honest.** A vote that could not be read is recorded as
  unparsed rather than scored; a numeric answer outside the scale is
  clamped and flagged.

The judge runs through `chat`, so it inherits the budget cap, concurrency,
resumability and per-call provenance; a local model is the cheap first test.
""",
    inputs=(
        In("records", "records/record",
           "The subjects to grade, each with `text` — or `text_a` and "
           "`text_b` for a pairwise scale. A document collection is read the "
           "same way.", many=True),
    ),
    emits=Emits('eval/verdict', collection=True, doc='One item per subject: `id`, `coords`, the verdict (`score`/`spread`/`min`/`max`, or `label`/`counts`/`agreement`, or `winner`/`counts`/`agreement`), `rationale`, `n_votes`, `n_parsed`, every `vote`, and `unparsed: true` when no vote could be read. The header carries `judge` (who graded and how), `summary` (mean/median/stdev or counts, `n_unparsed`, `first_shown_win_rate` for pairwise) and `spend`.'),
    params=(
        P("judge", "object",
          "Who grades: `{\"model\": …, \"system\": rubric, \"max_tokens\": "
          "512, \"temperature\": 0.0, \"budget_usd\": …, "
          "\"provider_options\": …}`. `model` is required; `system` is the "
          "rubric unless `rubric` is given."),
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
          {"type": "numeric", "min": 1, "max": 5}),
        P("n_votes", "int", "How many times each subject is judged.", 1),
        _BUDGET,
        P("concurrency", "int",
          "How many judge requests are in flight at once (remote judges).",
          4),
    ),
    example={
        "judge": {"model": {"provider": "anthropic", "model": "claude-sonnet-5"},
                  "system": "You grade short stories for originality."},
        "rubric": "1 = a stock plot told plainly; 5 = a premise you have not seen before.",
        "scale": {"type": "numeric", "min": 1, "max": 5},
        "n_votes": 3,
        "budget_usd": 3.0,
    },
    example_inputs={"records": {"$fetch": "$stories"}},
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
    emits=Emits('records/table', collection=False, doc='One row per value the metric returned: `metric`, `variant`, `value`, `n`.'),
    params=(
        P("metric", "string",
          "The hub metric's name: `\"accuracy\"`, `\"exact_match\"`, "
          "`\"f1\"`, `\"bleu\"`, `\"rouge\"`, …"),
        P("kwargs", "object",
          "Extra keyword arguments for the metric's `compute` — e.g. "
          "`{\"average\": \"macro\"}` for F1.",
          None),
        P("variant", "string",
          "A label for which model produced the predictions, stamped on "
          "every row as a coordinate.",
          "base"),
    ),
    example={"metric": "exact_match", "variant": "adapter"},
    example_inputs={"records": {"$fetch": "$answers"}},
)

EVAL_SUITE = Op(
    name="eval/benchmark",
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
    emits=(
        Emits('records/table', collection=False, doc='One row per (task, metric): `task`, `metric`, `variant`, `value`, `stderr`, `n`.')
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
    example={"model": "$model", "tasks": ["hellaswag", "arc_easy"], "limit": 200, "variant": "base"},
)

FINETUNE_LORA = Op(
    name="adapter/train",
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

With `depth` > 1 the outcome is a *sequence* of slots (a list of three
colours), each slot with its own target (`per_slot`) or all sharing one;
`join` and `closer` are the text between and after them. Sequences are
sampled fresh every step. The **naturalism gate** checks first that every
sampled sequence tokenizes to exactly one token per slot, so slot *i* is
token *i* — an adapter trained on a vocabulary the tokenizer splits would
be training on noise.

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
    emits=Emits('adapter/lora', collection=False, doc="`data` (safetensors bytes), `format`, `base_model`, `trained_on` (the base and any prior adapters), `lora` (rank, alpha, scale, target modules, parameter count) and `train` (steps, lr, seed, batch, final loss, the target spec, depth, positions, counts). Wire it into a later node's `adapter` port, or `adapter/publish`."),
    params=(
        P("target", "object",
          "The target distribution — `{\"uniform\": [...]}` or "
          "`{\"weights\": {...}}`, with optional `transform` steps, "
          "`depth`, `join`, `per_slot`. See above."),
        P("steps", "int", "Training steps.", 250),
        P("lr", "float", "Learning rate.", 1e-4),
        P("lora", "object",
          "`{\"rank\": 8, \"alpha\": 16, \"target_modules\": [\"q_proj\", "
          "\"v_proj\"]}` — the adapter's shape.",
          {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}),
        P("batch", "object",
          "Items per step by kind: depth 1 `{\"target\": 3, \"anchor\": 1, "
          "\"continuation\": 2}`; depth > 1 `{\"sequence\": 3, \"target\": 1, "
          "\"anchor\": 1}`.",
          None),
        P("closer", "string",
          "The text after the outcome that closes the decision — `\" }\"` "
          "for depth 1, `'\"'` for deeper tries.",
          None),
        P("positions", "\"all\" | \"skip_first\" | list[int]",
          "For depth > 1: which slots are trained. `\"skip_first\"` leaves "
          "the first slot untrained.",
          "all"),
        P("marginal", "bool",
          "For depth > 1: also train the first slot's marginal distribution "
          "as its own item. Turn off with `positions: \"skip_first\"`.",
          True),
        P("naturalism", "bool | object",
          "Run the one-token-per-slot gate before training; `{\"samples\": "
          "40}` sets how many sequences it checks.",
          True),
        P("checkpoint_every", "int",
          "Save resumable training state every this many steps.",
          50),
    ),
    example={
        "model": "$model",
        "target": {"weights": {"$fetch": "$frequencies"},
                   "transform": [{"op": "sqrt"}, {"op": "normalize"}]},
        "steps": 300,
        "lora": {"rank": 8, "alpha": 16},
        "seed": 7,
    },
    example_inputs={"records": {"$fetch": "$prompts"}, "anchors": {"$fetch": "$anchors"}},
)

HF_PUSH_ADAPTER = Op(
    name="adapter/publish",
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
    emits=(
        Emits('adapter/push', collection=False, doc='`repo`, `private`, `files` (name and size), `lora`, `base_model`, `commit`, `url`, and `hf_adapter_ref` to fetch it back.')
    ),
    params=(
        P("repo", "string", "The destination, `\"<namespace>/<name>\"`."),
        P("private", "bool", "Create the repository private.", True),
        P("commit_message", "string", "The commit message.", "mechbench: adapter push"),
    ),
    example={"repo": "benjismith/spinner-fair-v1", "private": True},
    example_inputs={"adapter": {"$fetch": "$adapter"}},
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
    emits=Emits('adapter/checkpoint', collection=False, doc='Where the checkpoint landed, its files with their hashes, and the stack that was merged.'),
    params=(
        P("to", "object",
          "Where to publish: `{\"bench\": {\"name\": …}}` (lowercase, "
          "digits, `-`, `_`) or `{\"hf\": {\"repo\": …, \"private\": …}}`. "
          "Exactly one."),
    ),
    example={"model": "$model", "to": {"bench": {"name": "spinner-fair-v1"}}},
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
    emits=None,
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
    emits=None,
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
    HF_PUSH_ADAPTER, MERGE, TOOLS_CALC, TOOLS_BENCH_LOOKUP,
)
