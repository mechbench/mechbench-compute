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
