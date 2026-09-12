# mechbench-compute — changelog

Every release carries two lists, and an empty one says `_None._` rather
than being omitted: "there were none" and "nobody thought about it"
must not look the same. The convention and the reasoning are in
[`mechbench/docs/RELEASE_NOTES.md`](https://github.com/mechbench/mechbench/blob/main/docs/RELEASE_NOTES.md).

The entries below for 0.30.0–0.40.0 were written **retroactively** on
2026-09-11, after the convention was adopted (task 000434). They span
experiment 024's entire run, which is why the exercise was worth
doing: three of those releases changed numbers, and at the time
nothing said so.

---

## 0.49.0 — 2026-09-11

### Changes that raise

- `cache: true` on a chat node run from a bare `ProtocolSpec` with no
  `protocolId` still refuses and asks for a name — there is no stable
  identity to derive one from. Inside a job there always is.

### Changes that alter results without raising

- **`cache: true` now works, deriving its memo label from the
  protocol id and the node id** — `<owner>/memos/<protocolId>/<nodeId>`.
  0.46.0 refused it and demanded an explicit name, arguing that a
  derived label would be discarded by every compute release. That was
  true of a label derived from the node FINGERPRINT; the protocol id
  and node id are stable across releases and the job spec already
  carries them. Reconsidered: refusing was friction for no gain.

---

## 0.48.0 — 2026-09-11

### Changes that raise

- _None._

### Changes that alter results without raising

- **`Snapshot.to_wire()` no longer inlines blobs by default.**
  Measured on a 2000-file tree: 8.43 MB inline against 0.22 MB as
  hash references — 38x. A sandbox session emits one snapshot per tool
  call, so inlining put the whole tree on the wire for every `ls`.
  `to_wire(inline=True)` keeps the self-contained form for fixtures.
  The DIGEST is unchanged either way: inlining is a storage decision,
  not a fact about the tree.
- **Read-only mounts are excluded from capture.** `Mount(at, object,
  digest)` marks a tree that cannot have changed; capture skips it and
  the digest folds it in by IDENTITY. Mounting a 200 MB corpus was
  costing a full re-hash on every tool call and buying nothing.

---

## 0.47.0 — 2026-09-11

Filesystem snapshots (task 000358, the base of the sandbox arc).

### Changes that raise

- _None._ New module: `mechbench_compute.snapshots`.

### Changes that alter results without raising

- _None._ Nothing consumes it yet.

`Snapshot` is a directory as a content-addressed value, so a sandbox
tool call can be `(snapshot, argv) -> (snapshot', stdout, stderr,
exit)` — an ordinary item with lineage instead of a directory somebody
mutated. `capture` / `materialize` / `diff` / `seeded`, with limits on
file count and total bytes, and a symlink leaving the root refused
rather than silently resolved.

What the digest deliberately ignores: mtimes, ownership, real
permission bits, and the order the OS returned entries in. What it
keeps: paths, content hashes, and the executable bit — which changes
what a later run DOES.

Entries sort at construction rather than in each constructor. `os.walk`
is depth-first, so `capture` produced `a.txt, z.txt, m/q.txt` while
`from_wire` produced sorted order, and the digest walks entries — so
the same tree hashed two different ways depending on how it was built.
Caught by the test that asserts exactly that.

---

## 0.46.0 — 2026-09-11

Memoized remote calls (task 000355, epic 000334).

### Changes that raise

- **`cache: true` refuses** — a memo needs a label to live under.
  `cache: "<owner>/<project>/memos/<name>"`. A memo keyed on node
  identity would be thrown away by every compute release, which is
  backwards: the compute version is not part of what a provider was
  asked, and the request hash inside the memo is what decides a hit.
- **The `chat` block now declares its params** (000438), so a typo or
  an unsupported option is refused by name rather than ignored.

### Changes that alter results without raising

- **A replayed call costs nothing.** `resp.replayed` settles at $0 and
  releases its reservation instead of charging it — a cached re-run
  was otherwise billing for a purchase it did not make. The original
  call's token usage is KEPT on the record, because comparing a
  memoized run against its first run needs it.
- With `cache` set, a chat node loads its memo, runs the transport in
  `auto` mode, and writes the memo back. The node's summary gains
  `cache: {label, hits, recorded, entries}`.

Also: `tests/test_block_params.py` reads each declared block's source
and asserts every param it reads is declared. An incomplete
declaration is a false refusal — the opposite bug from the one
declaring params was meant to fix — and the table can no longer drift
behind the code.

---

## 0.45.0 — 2026-09-11

### Changes that raise

- _None._

### Changes that alter results without raising

- **A model's output is truncated at its first tool call.** Anything
  written after the call's closing marker is the model **fabricating
  the tool response** rather than waiting for it. Observed verbatim on
  024's P2, where gemma-4-e2b wrote its call and then invented the
  answer:

      <|tool_call>call:calc{expression:<|"|>37 + 18<|"|>}<tool_call|>
      <|tool_response>response:calc{value:<|"|>55<|"|>}<tool_response|>

  Keeping that text put a fabricated response in the transcript beside
  the real one, and the following turn came back empty — 79 of 80
  items in every arm. The reasoning BEFORE the call is genuine and is
  kept.

  With this, the tool loop completes: call, execute, answer. Verified
  against the real model — 55, 77 and 117 on three arithmetic items,
  where every previous batch returned empty.

---

## 0.44.0 — 2026-09-11

### Changes that raise

- _None._

### Changes that alter results without raising

- **A parsed tool call is removed from the assistant turn's text.**
  The call goes back into the transcript as a structured `tool_calls`
  entry, which the template renders in the model's own format; leaving
  the raw markup in the message content too put the call in the
  transcript **twice**, and a model handed its own call twice answers
  with nothing. Observed on 024's P2: every arm executed its call
  correctly and then returned an empty final turn.

  `ToolDialect.parse` returns `(text, calls)` now, restoring a
  contract the deleted `parse_tool_calls` had and I dropped.

- **A call to a tool that was never offered stays in the text.** It is
  not executed and not stripped — stripping it would erase the only
  evidence of what the model tried, which `tools.errors` needs to
  report `unknown_tool` with the name.

---

## 0.43.0 — 2026-09-11

### Changes that raise

- **`on_tool_error: "fail"`** makes a tool-call failure fatal for the
  node. The default is `"record"` — an individual failed call does not
  fail a run — and the name mirrors group-stats' `on_missing` rather
  than inventing a second idiom for the same choice.
- `FAMILIES`, `render_tools`, `parse_tool_calls` and
  `looks_like_a_tool_call` are **deleted**. The tool protocol comes
  from the model's chat template now (0.42.0); two ways to read a tool
  call is how the next reader picks the wrong one.

### Changes that alter results without raising

- **`tool_near_misses` is gone, replaced by `tools.errors`.** Benji:
  "It doesn't matter whether a miss is near or not. It's an error."
  Correct — and the old name described how close the model got, which
  is neither well-defined nor actionable.

  The node now reports `tools: {dialect, responses, with_calls,
  without_calls, errors, errors_by_cause}`, and each item carries its
  own `tool_errors` so a failure can be sliced by the condition that
  produced it. Causes: `unknown_tool`, `unparseable_call`,
  `no_dialect`, `execution_failed` — the last of which was previously
  visible only per-item in `tool_runs` and never aggregated.

  **Answering without calling a tool is NOT an error**, and is counted
  rather than faulted. Whether the model should have called one is the
  experiment's question, not the harness's.

---

## 0.42.0 — 2026-09-11

Tool dialects taken from each model's own chat template (epic 000439).
The harness used to invent a markdown-fence convention and ask every
local model to speak it; models speak the protocol they were trained
on, which ships in `chat_template.jinja` beside the weights.

### Changes that raise

- **Offering `tools` to a model with no tool protocol now raises
  `NoToolDialect`** instead of falling back to our fence. `gemma-3` is
  a live example: its template accepts `tools=` and renders the same
  prompt either way. A model that cannot receive a declaration
  produces output indistinguishable from a model that chose not to
  call anything, which is how experiment 024 lost an arm.
- A model whose template declares tools but matches no known dialect
  also raises, showing its rendering, rather than guessing.

### Changes that alter results without raising

- **Tools are declared by the model's own template**, not by a system
  prompt fragment we wrote. Every prompt containing tools changes.
- **Tool calls are parsed per dialect** — gemma-4's
  `<|tool_call>call:name{k:<|"|>v<|"|>}`, qwen-2.5's
  `<tool_call>{json}</tool_call>`, llama-3's bare `{"name",
  "parameters"}`. Calls that previously went unread now execute.
- **Results go back as real tool turns under the right role** — Llama
  reads them as `ipython`, Qwen and Gemma as `tool` — rather than
  being stringified into prose. The model reading its own tool output
  as if a user had said it was the third leg of the same bug.
- `tool_near_misses` now carries `tool_miss_reasons`
  (`unknown_tool` / `unparseable_arguments` / `wrong_envelope` /
  `no_dialect`), samples, and the resolved `tool_dialect`.

Every dialect is pinned by a **round-trip conformance test**: render a
canonical call through the model's own template, parse it back, assert
equality. Fixtures run everywhere; the live version runs under
`MECHBENCH_MODEL_TESTS=1` against the real tokenizers, so a model
publishing a new template fails a test instead of an experiment.

---

## 0.41.0 — 2026-09-11

### Changes that raise

- _None._

### Changes that alter results without raising

- **A bare argument in a native tool call now binds to the tool's one
  required parameter.** `<|tool_call>call:calc(37 + 18)` parses as
  `{"expression": "37 + 18"}` where it previously parsed as nothing.
  More tool calls execute; runs with `tools` on a local model can
  produce different output.

  Ambiguity is still refused: a bare argument for a tool with two
  required parameters is a guess, and this does not guess.

  Found within minutes of shipping 0.40.0, because the rendered
  instruction changed in that release and gemma-4-e2b changed format
  in response — and `tool_near_misses` reported 80 of 80 instead of
  the silence that hid the same class of problem for 320 generations.

---

## 0.40.0 — 2026-09-11

### Changes that raise

- **A block refuses a param it does not read** (000438).
  `block_params.check_params` runs before dispatch. Declared per block
  and opt-in: today `vectors/mst` and `residuals/vectors`; every other
  block is unchecked exactly as before. A protocol newer than the
  runner now fails with a message naming the param rather than
  quietly producing something else.

### Changes that alter results without raising

- **Gemma's native tool calls are parsed** (000437). `family: "gemma"`
  previously matched only a fenced ```` ```tool_code ```` block.
  `<|tool_call>call:<tool>({…})` now parses too — so **tool calls that
  silently did nothing will now execute**. Any run with `tools` on a
  local Gemma model can produce different output; experiment 024's P2
  is being re-run for exactly this reason. The rendered instruction
  also changed, which changes the prompt, which changes sampling.
- `render_tools` output gained a line, so any cached completion keyed
  on the rendered system prompt misses.

Also new, and the reason the list above is possible to write:
`tool_near_misses` on the chat block's summary counts responses that
were reaching for a tool and produced no call. Reported even at zero.

---

## 0.39.0 — 2026-09-11

### Changes that raise

- `vectors/mst` with `center: true` and a `similarity_matrix` input
  **refuses** — the vectors are gone by then and centering is
  impossible. Silently not centering was the alternative.

### Changes that alter results without raising

- _None._ `center` defaults to **off**, so every stored result keeps
  its numbers. This was deliberate: the correction that motivated it
  (experiment 024's variety ranking) is recorded as a retraction in
  the experiment, not as a silent change under existing protocols.

---

## 0.38.0 — 2026-09-11

### Changes that raise

- _None._

### Changes that alter results without raising

- **`__version__` gained a source digest on non-installed imports**
  (000433): a package imported from outside `site-packages` reports
  `0.38.0+src.<12 hex>`. `core_version` feeds `node_fingerprint`, so
  **every node fingerprint changes on a dev machine, and changes again
  whenever any compute source file changes.** No computed number
  changes; what changes is that resume partials stop being reused
  across edited code, which is the point. Released wheels are
  unaffected and keep the bare version.

---

## 0.37.0 — 2026-09-11

### Changes that raise

- `pool: "last_k"` without a positive integer `pool_k` refuses, as
  does an unknown pool or a negative `pool_skip`.

### Changes that alter results without raising

- _None,_ and there is a test asserting it: `residuals/vectors`
  records made without `pool` are unchanged in every field. The new
  parameter only acts when asked for.

---

## 0.36.0 — 2026-09-10

### Changes that raise

- **A completion with output tokens but no text now raises.** The
  Anthropic adapter was silently dropping content blocks it did not
  recognize, yielding an empty string; five of experiment 024's 200
  frontier stories were empty this way. It now collects unmapped
  blocks and refuses, and `CallRecord` carries `stop_reason`.

### Changes that alter results without raising

- **`skip_empty` on `residuals/vectors`** drops records with no text
  when set. It changes `n`, and therefore every downstream statistic.
  The dropped ids are reported on the output, and it is off by
  default.

---

## 0.35.0 — 2026-09-10

### Changes that raise

- _None._

### Changes that alter results without raising

- **A `document_collection` is a record stream everywhere.** Blocks
  that previously refused one ("input is not a record stream") now
  accept it. Protocols that were failing will now run — which is the
  good case, but a graph that *depended* on that refusal behaves
  differently.

---

## 0.34.0 — 2026-09-10

### Changes that raise

- _None._

### Changes that alter results without raising

- _None._ New block only: `~canonical/ops/vectors/mst/1`.

---

## 0.33.0 — 2026-09-10

### Changes that raise

- _None._

### Changes that alter results without raising

- **`http.client.HTTPException` and `OSError` are now retried** rather
  than escaping as fatal. A run that would have died mid-corpus now
  completes — with the retried items sampled at a different point in
  the stream. Found when a `RemoteDisconnected` killed experiment
  024's P1 at 293/601.

---

## 0.32.0 — 2026-09-10

### Changes that raise

- Limiter exhaustion raises `ProviderUnavailable` instead of
  `RuntimeError`, so the runner reports an **interrupted** job rather
  than a failed one, and the work already paid for stays resumable.

### Changes that alter results without raising

- **The rate limiter queues (FIFO tickets) and can raise its own
  capacity** from `*-tokens-limit` headers. Concurrent items are
  therefore ordered differently than before, and a run that previously
  starved now finishes. Anything order-sensitive downstream sees a
  different order. Found when 154/200 items of a $0.69 run were lost
  to starvation.

---

## 0.31.0 — 2026-09-09

### Changes that raise

- `group-stats` with `on_missing: "error"` (the default) raises on a
  row missing the value field, where it previously threw a bare
  `KeyError` from inside the block.

### Changes that alter results without raising

- **`on_missing: "skip"` excludes rows from the aggregate** and
  reports `n_missing`. A judged corpus with unparsed rows now produces
  a mean over fewer rows instead of failing. Off by default.

---

## 0.30.0 — 2026-09-09

### Changes that raise

- _None._

### Changes that alter results without raising

- **A judged corpus keeps its coords** (000356 follow-on): `coords_of`
  reads them from `metadata` as well as the top level, so grouping a
  judged stored corpus by its condition now works where it previously
  produced a single undifferentiated group. Any `group-stats` over a
  judge's output changes shape.

---

## Before 0.30.0

Not classified. The convention was adopted at 000434 and applied
backwards only as far as experiment 024's run, where the numbers were
still live enough to audit honestly.
