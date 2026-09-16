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

## 0.90.0 — 2026-09-17

### Changes that raise

- _None._ This releases a graph that used to raise and should not have.

### Changes that alter results without raising

- **A placeholder join no longer fails the run it was keeping alive.**
  Every node's result is emitted with its upstreams' object paths as
  lineage; a node running under `on_missing: "skip"` or `"placeholder"`
  (000399) has an upstream that produced nothing and therefore stored
  nothing, and citing that absence as a path raised `KeyError` — so the
  whole join policy worked in-process and died the moment a real job
  stored its results. Lineage now names the inputs that exist;
  `nodes_missing` on the manifest is where the absence is recorded.
  Found by the first STORED run of one (task 000515); the tests that
  covered the policy all handed the graph straight to the executor,
  which stores nothing, and one of them does not any more.

### Other

- _None._

## 0.89.0 — 2026-09-17

### Changes that raise

- _None._

### Changes that alter results without raising

- _None._

### Other

- **`bench.watch` stops on `done_with_missing`** (task 000515). A run
  that finished having lost a branch is finished; without this the
  watcher polls it forever, which is the shape of every bug a new
  terminal status causes.

## 0.88.0 — 2026-09-17

One bug, found by reading the votes behind a verdict the demonstration
protocol produced.

### Changes that raise

- _None._

### Changes that alter results without raising

- **A pairwise verdict is mapped back from what the judge saw**
  (`eval/judge`). The A/B order flips per vote, and the judge's answer
  was counted as the letter it gave — never mapped back to the record's
  own `text_a`/`text_b`. So the randomisation that was supposed to make
  a pairwise result trustworthy scrambled it instead: a judge that
  chose the same passage in all three votes was reported as a 2–1 split
  for the OTHER passage, with `agreement` 0.67 rather than 1.0, and
  `first_shown_win_rate` — the diagnostic for exactly this — was
  computed from unmapped labels, reading 0.0 for a judge with no
  position bias at all. **Every pairwise verdict from every earlier
  release is wrong** wherever votes saw different orders (any
  `n_votes > 1`, and half of all single votes). Re-run them. Each vote
  now carries `shown_winner` beside the mapped `winner`, and a row
  whose rationale was written under a swapped order says so with
  `rationale_order: "BA"` — the letters in that text are the judge's,
  not the record's.

### Other

- _None._

## 0.87.0 — 2026-09-17

What writing the first two protocols against the dataflow set turned up.
Each of these is reachable only now that a branch can go missing and
leave the run standing.

### Changes that raise

- **A judge is refused a subject with nothing to read** (`eval/judge`).
  A record whose judged field — `text`, or `text_a`/`text_b` on a
  pairwise scale — is absent or blank now fails the node by name, where
  before the empty string was shown to the judge and the answer scored.
  A winner over an empty string is indistinguishable from any other
  winner in the column, and this became reachable the moment
  `records/zip` gained `on_missing: "placeholder"`: the failed branch
  keeps its key and its side arrives empty. `on_missing: "skip"` keeps
  those subjects as `unjudged` rows naming what was missing, and grades
  the rest; the summary carries `n_unjudged` and which.

### Changes that alter results without raising

- **A judge sends no `temperature` unless one is named** (`eval/judge`).
  The block sent `0.0` for every judge that did not say otherwise,
  which made `claude-sonnet-5` unusable as a judge outright — it
  answers HTTP 400, "`temperature` is deprecated for this model". A
  default the provider may refuse is not a safe default; a judge's
  steadiness is bought with `n_votes` and reported as `agreement`.
  Scores from a remote judge that never named a temperature will move,
  because the request now takes the provider's own default. Name
  `judge: {"temperature": 0.0}` to keep the old requests exactly.

### Other

- **A `records/map` body sees the protocol's own bindings** (task
  000400). The body is a sub-protocol, not a foreign graph: a run
  launched with `$model` can now name it inside the body instead of
  carrying the same constant on every record to bind it in. `bind`
  shadows them, since the per-record value is the specific one.

- **The mock provider can refuse from inside a graph.**
  `provider_options: {"mock": {"fail": "…"}}` raises, the way `text`
  fixes the reply — so a protocol author can rehearse the branch that
  goes down, and check the placeholder path, before a real provider
  proves it at an awkward moment. Previously only a caller that
  constructed the transport could (`script=`), which a graph cannot.

## 0.86.0 — 2026-09-17

The rest of the dataflow set: what happens when a branch fails, when two
branches could run at once, and when a whole sub-protocol should run per
record.

### Changes that raise

- _None._ A graph that declares no `on_missing` policy fails exactly as
  it did, and a graph with no independent remote nodes runs exactly as
  it did.

### Changes that alter results without raising

- **A failed node's siblings finish before the run fails** (task
  000399). The failure is raised when the run is otherwise over rather
  than at the moment it happens, so a branch that was going to succeed
  still does — which is also what makes the scheduler below safe. The
  error a job fails with is unchanged.

### Other

- **`on_missing` on a port** (000399): `fail` (the default, and what
  every graph did), `skip` (this node is skipped too, and its own
  consumers see it missing in turn) or `placeholder` (the port gets its
  kind's empty value carrying a `missing` marker). The OP declares what
  its port can meaningfully do without the input; an EDGE may override,
  because whether a partial result is worth having is a question about
  the experiment. `placeholder` on a port that takes one object is
  refused at load — an empty collection is a real value, an empty
  `direction/vector` is not. The result's `nodes_missing` says what did
  not run and why, so an absence is never inferred from a shorter list.

- **Independent remote nodes run at once** (task 000396). When the
  executor reaches a node whose work is a provider's, every other such
  node whose inputs are already computed goes with it — two prompts to
  two providers, then a judge, now takes one call's latency instead of
  two. Deliberately narrow: a local model node must serialize (one model
  in memory, one fused adapter at a time) and a pure block takes
  microseconds, where a thread would be risk without a gain. At most
  eight nodes in flight, under the provider limiter that already bounds
  the requests within each. Every result is hashed, emitted and counted
  on the calling thread in topological order, so the manifest and the
  stored objects are identical to a serial run; item spooling is bound
  to its own node, so a resumed job cannot reuse another node's work.

- **`records/map`** (task 000400): run a whole sub-protocol once per
  record. A run set fans out over runs and a node fans out over its own
  items; between them there was nothing. `body` is a graph, run per
  record with that record's fields bound into its holes by `bind`; each
  invocation is an item keyed by the record's id, so an interrupted map
  resumes the way an interrupted chat node does. `collect` is `stream`
  (flatten every invocation), `first` or `all`. A body by REFERENCE to a
  stored protocol waits on task 000393; an inline body works now.

## 0.85.0 — 2026-09-17

### Changes that raise

- **Two edges into one port are refused** (task 000397). A port that
  takes one input and was wired twice kept whichever edge came LAST in
  the graph's edge list — silently, and which one that was depended on
  the order the author happened to write the lines in. The graph now
  says so at load, naming the node and the port, and says what used to
  happen.

### Changes that alter results without raising

- **A node's in-edges are read in a canonical order**: by port, then by
  the edge's own `index`, then by the source node's id. Its inputs, its
  lineage and its fingerprint all read that order, so **moving an edge
  in the protocol's JSON no longer changes a fingerprint** — before
  this, reordering the edge list restarted every cached and resumed
  thing downstream of it. The one-time cost is that a node with more
  than one in-edge whose list was not already in this order gets a new
  fingerprint, and its cached work starts over. No number changes.

### Other

- **Variadic ports** (task 000397): a port may declare that it collects
  SEVERAL edges rather than one — `In(…, variadic=True, min_edges=2)` —
  and the block receives them in order as `[{node, value}, …]`, so it
  knows which branch each came from. `many` and `variadic` are
  different and compose: `many` is one value that is a collection,
  `variadic` is several edges.

  `records/union` keeps its wildcard port rather than gaining a
  variadic form: its port NAMES are load-bearing (they become the batch
  coordinate every union-then-compare protocol groups on), and two ways
  to say the same thing would be one too many.

- **`records/zip`** (task 000398): align several branches' records into
  one record per key — record 7 of each branch together — keeping which
  branch each came from. `union` concatenates and marks the source;
  this pairs. Key on `id` or on named coordinates (`by: ["prompt",
  "seed"]`) for branches that number their records differently but
  share a design. A key missing from some branch fails by default,
  because a silently shorter output is a silently different experiment;
  `drop` keeps what every branch has, `placeholder` keeps them all and
  marks what is absent. Branches are named by their source node unless
  `names` says otherwise, and `flatten` lifts each branch's fields under
  a `<branch>_` prefix.

## 0.84.0 — 2026-09-17

### Changes that raise

- _None._

### Changes that alter results without raising

- _None._

### Other

- **`intervene/apply` edits weights, not only activations** (task
  000457, the write half). A spec item that names a **`parameter`**
  instead of a `point` edits the model itself:

  ```json
  {"parameter": "layers.12.self_attn.o_proj", "op": "project_out",
   "direction": {"$fetch": "$axis"}}
  ```

  The two kinds compose in one spec and differ in scope: an activation
  edit lasts for one forward pass, a weight edit for the node. Four ops
  — `zero`, `scale`, `project_out` (take a direction out of what a
  module reads or writes, the side chosen per module), `truncate` (keep
  the top `rank` singular directions). `parameter` takes the same names
  `weights/capture` does, `*` included, so one item can zero every
  layer's `o_proj`.

  **The model is put back exactly.** The original tensors are kept and
  reinstalled — never a subtraction that would not round-trip in bf16 —
  and the restore happens in a `finally`, so a forward pass that raises
  cannot leave a later node running against an edited model. Verified
  against Gemma 4 E2B: every `o_proj` scaled to zero moves the readout
  from certain (H = 0.000) to noise (H = 6.864), and the weights come
  back bit-identical. The readout's header carries a `weights` list
  beside `spec`, so a reader knows the model was not the one on the
  shelf.

  A sweep re-applies the edits at each factor, so `{"strength": [0.5,
  1.0]}` on a `project_out` removes half a direction and then all of
  it; factor 0 is the unedited model.

## 0.83.0 — 2026-09-17

### Changes that raise

- _None._

### Changes that alter results without raising

- _None._

### Other

- **A `weights` family: the model's parameters, read as data** (task
  000457). Every other family reads a forward pass; this one reads the
  learned matrices themselves, with no prompt and nothing to be
  representative of.

  - **`weights/capture`** names parameter points the way the module tree
    does — `layers.12.self_attn.q_proj`, `embed_tokens`,
    `layers.*.mlp.down_proj`, with `*` standing for one segment — and
    emits one **`weights/parameter`** per tensor: shape, `frobenius`,
    `mean`, `std`, `max_abs` (where an outlier channel shows) and
    `sparsity`. `spectrum: k` adds the top singular values, σ₁ and the
    effective rank at the cost of an SVD per tensor; `values: true`
    carries the tensor itself and is refused past two million numbers.
    The cheap stats are computed where the tensor is, in row blocks with
    the sums accumulated in float64: all 4.6 billion parameters of
    Gemma 4 E2B in **1.8 seconds**, and more accurately than a float32
    pass over the host copy, which was also 20× slower.
  - **`weights/decompose`** turns a parameter's principal directions
    into `direction/vector` items — in the residual stream, which is the
    only space the rest of the platform can talk about. `q_proj`,
    `k_proj`, `v_proj`, `gate_proj` and `up_proj` READ the residual
    stream, so their right singular vectors are the directions they ask
    about; `o_proj` and `down_proj` WRITE it, so their left singular
    vectors are what they contribute. A module where neither side is the
    residual stream is refused by name. What comes out is the same kind
    `direction/fit` emits from activations, so `direction/unembed` names
    the tokens a weight direction promotes and `geometry/compare`
    measures a weight direction against an activation one — which is the
    bridge between the two halves of the grammar.

  The parameter-scoped `intervene` from the same task (editing a weight
  for the life of a run, with the fuse/restore contract) is not in this
  release.

## 0.82.1 — 2026-09-17

### Changes that raise

- **A protocol is proved runnable before it runs** (task 000513). The
  executor checked a node's params and ports when execution reached it,
  so a graph that could not run spent everything upstream of the first
  mistake proving so: one 014 trace took five resumes and most of a day
  to arrive at `activations/capture does not accept 'template'`, a
  refusal that was decidable at load. Now, for every node, before any
  node runs: the operation exists, every param is one the op reads, and
  every port an edge or an `inputs` entry names exists with the required
  ones filled. **Every problem is reported together**, not the first —
  a protocol being carried forward usually has several, and
  fix-one-run-again over a long protocol is the expensive version of
  this bug.

  No graph that ran before fails now: these are the same checks, at the
  same strictness, earlier. What a port is filled WITH is still checked
  in the loop, because that is the upstream node's output and is not
  knowable at load.

  Stored protocols written before typed ports (0.78.0) put a port's
  value under `params`; 0.82.0 stopped lifting those onto the port, so
  they already refused — this release refuses them at load, naming
  every node at once. Re-authoring is the fix, and an author script
  that emits the current shape produces a protocol that runs.

### Changes that alter results without raising

- _None._

## 0.82.0 — 2026-09-16

The release the alias tables named. Both of them — the 2026-09-14 family
renames and the 2026-09-16 verbs — are gone, along with the lift that
accepted an input under `params`. Everything below has been resolving
with a `RetiredOpName` warning that named this version; the protocols
stored on the bench were migrated on 2026-09-16 and a dry run the same
evening found nothing left to rewrite.

### Changes that raise

- **Sixty-six retired spellings are refused.** Not silently: the
  refusal names the current spelling, so a protocol carried forward from
  an old one is a one-line fix rather than a search.

| Refused | Write |
|---|---|
| `activations/vectors`, `residuals/vectors` | `activations/capture` |
| `activations/attention`, `attention/patterns` | `activations/capture-attention` |
| `activations/divergence`, `residuals/divergence` | `activations/contrast` |
| `merge` | `adapter/merge` |
| `hf/push-adapter` | `adapter/publish` |
| `finetune/lora` | `adapter/train` |
| `direction/from-pca` | `direction/decompose` |
| `direction/from-vectors` | `direction/fit` |
| `direction/vocab` | `direction/unembed` |
| `eval/suite` | `eval/benchmark` |
| `eval/expectation` | `eval/expect` |
| `judge` | `eval/judge` |
| `eval/hf-metric`, `eval/metric` | `eval/score` |
| `direction/similarity`, `geometry/similarity`, `vectors/similarity` | `geometry/compare` |
| `geometry/mst`, `vectors/mst` | `geometry/span` |
| `ablate/heads`, `intervene/heads` | `intervene/ablate-heads` |
| `ablate/layers`, `intervene/layers` | `intervene/ablate-layers` |
| `intervene` | `intervene/apply` |
| `intervene/trace`, `patch/trace` | `intervene/patch` |
| `steer/inject` | `intervene/steer` |
| `attribution/logits`, `logits/attribution` | `logits/attribute` |
| `decision-read`, `logits/decision` | `logits/read` |
| `lens-trajectory`, `logits/funnel` | `logits/read-layers` |
| `lens/positions`, `logits/lens` | `logits/scan` |
| `records/histogram`, `reduce/histogram` | `records/bin` |
| `factor-cross`, `grid` | `records/cross` |
| `records/template`, `template` | `records/fill` |
| `records/chart`, `viz/spec` | `records/plot` |
| `records/top-k`, `reduce/top-k` | `records/rank` |
| `select` | `records/select` |
| `paired-delta`, `records/delta` | `records/subtract` |
| `group-stats`, `records/stats` | `records/summarize` |
| `records/table`, `table/from-records` | `records/tabulate` |
| `records/sum`, `reduce/sum` | `records/total` |
| `union` | `records/union` |
| `chat` | `text/chat` |
| `conversation`, `text/conversation` | `text/converse` |
| `generate` | `text/generate` |
| `text/stats` | `text/measure` |
| `score` | `text/score` |
| `tokenize/stats` | `text/tokenize` |
| `tools/bench-lookup` | `tools/lookup` |

  The `/1` version segment stored protocols carried is refused the same
  way (`records/select/1` → write `records/select`), and so is the
  stored path with a retired leaf. `lexicon.ALIASES` is now
  `lexicon.RETIRED`, read by `lexicon.explain_unknown` and by nothing
  else — no lookup resolves through it.

- **An input given under `params` is refused as an unknown param**, as
  the 0.78.0 notes promised. `records: {"$fetch": …}` belongs under
  the node's `inputs`; `check_params` names the port.

- **A graph is checked before it runs.** Every node's operation is
  resolved at load and ALL the bad ones are reported together, rather
  than each being discovered when execution reached it. A protocol whose
  last node is misspelled now fails in the first second instead of
  after everything upstream of it has been computed — which is how one
  014 trace spent five resumes and most of a day. (Params follow in the
  next release, task 000513.)

### Changes that alter results without raising

- _None._

## 0.81.2 — 2026-09-16

A patch again, for the same reason 0.81.1 was one: **0.82.0 is spoken
for** by the removal of both alias tables. Nothing here changes an
existing number; it adds a readout and makes two collections keep the
kind they already were.

### Changes that raise

- _None._

### Changes that alter results without raising

- **`records/union` and `records/select` keep the item kind.** A union
  of collections that share one item kind was labelled
  `records/record`, and so was a filtered selection — so a collection
  of adapter deltas or verdicts lost, at the first reshaping node, the
  kind whose metrics make it comparable. Both now carry the kind
  through when it is unambiguous (a union of different kinds, or a
  selection that projects `fields`, still lands on the root). **A
  stored union or selection changes `item_kind` and `key`, so its
  content hash moves**; the items and their numbers do not.

### Other

- **`adapter/measure`: what training wrote, read from the adapter**
  (task 000458). One `adapter/delta` item per (layer, module) — the
  Frobenius norm, the spectrum, the effective rank and the share of the
  adapter's mass — computed from the low-rank factors themselves, with
  no model, no prompt and no forward pass. The spectrum is exact rather
  than estimated: `ΔW = scale · B · A` has at most `rank` non-zero
  directions, so its singular values are those of an r×r matrix and the
  delta is never formed. With `vectors: true` each item carries its
  principal direction, so `geometry/compare` with `by: "module"`
  answers whether two training runs moved the model the same way — the
  weight-space form of an activation-space question.

## 0.81.1 — 2026-09-16

A patch, not a minor, only because **0.82.0 is spoken for**: the 0.80.0
notes promised both alias tables would be removed there, and the
`RetiredOpName` warnings in the wild name that version. Nothing that
worked before this release stops working — every refusal below replaces
a silent no-op.

### Changes that raise

- **`text/chat` on local weights refuses `json_mode`, `logprobs` and
  `tool_choice`** (task 000509). The local path dropped all three
  without a word: a protocol that asked a local model for JSON got
  whatever the model felt like, and nothing said the request went
  nowhere. Each is now refused by name, with what to do instead. The
  remote path is unchanged — those three have always been refused there
  by capability, per provider (`providers.base.check_supported`).
- **`tool_choice` with no `tools` is refused.** A choice among nothing:
  every adapter put it on the wire, where the provider ignored it or
  refused it in its own vocabulary.
- **A `provider_options` key that names no provider is refused.** The
  bag is keyed by provider — `{"anthropic": {"thinking": …}}` — so a
  field written at the top level was passed to nobody, which is the
  failure `provider_options` is most likely to produce.

### Changes that alter results without raising

- **`stop` works on the local path** (task 000509). `text/chat`
  declared it, every remote adapter sent it, and the local sampler read
  only the tokenizer's turn-end tokens — so a local node that asked to
  stop at a marker ran to `max_tokens` instead. It now ends the sample
  at the first marker, which is not part of the reply, as every
  provider's `stop` means. **A local chat node that sets `stop` will
  produce shorter text than it did.** `generate.sample_completion_cached`
  takes `stop_strings` for the same reason.
- **`platform_kinds.register_all` registers a NEW VERSION of a kind
  whose declared fields have changed**, instead of printing "pinned"
  and leaving the registry describing the fields the kind used to have
  (API task 000508). The registry refuses a changed manifest at a
  registered version and names the version it holds; `register_all`
  re-registers at the next number, and the manifest it replaces stays
  readable at `GET /kinds/<path>?version=N`. Needs an API from
  2026-09-16 or later; against an older one the refusal carries no
  version and is raised as before.
- **`BenchError` carries `status` and `body`.** A refusal's `code` is
  readable without matching on prose: `e.code() == "MANIFEST_PINNED"`.
  The message is unchanged.

## 0.81.0 — 2026-09-16

### Changes that raise

- _None._

### Changes that alter results without raising

- **`adapter/train` is repeatable: the same seed now trains a
  byte-identical adapter** (task 000507). The adapters' initial `A`
  matrices were drawn from MLX's global generator at `apply_lora` time,
  before the loop seeded anything. `B` starts at zero, so that draw
  changes nothing at step 0 and every gradient after it: two runs of the
  same protocol, same seed, same machine, same data produced different
  adapters — on experiment 002's eval battery, KL differing by up to
  0.19 bits and male-name entropy by 0.21, an envelope wider than the
  0.01-bit changes the experiments call drift when they compare
  releases. Each wrapped projection now draws from its own subkey of
  `mx.random.key(seed)`, in layer order, and the process's global
  generator is left alone. `apply_lora(..., seed=None)` keeps the old
  behaviour for a caller that wants an unrepeatable draw; the block
  always passes its `seed`.

  **An adapter trained before this release cannot be reproduced by
  re-running its protocol** — its initial draw is not recorded anywhere.
  Adapters already trained are unaffected as objects; only a re-run
  differs, and from here two re-runs agree with each other. The
  block's resume level is unchanged (`state-restorable`): that level
  describes how a partial is reused, not whether two runs agree.

## 0.80.0 — 2026-09-16

### Changes that raise

Nothing new raises in this release: every retired spelling below
resolves with a `RetiredOpName` warning, as the family renames did.
**The removal of both alias tables moves to 0.82.0** (it was 0.80.0 for
the first), so one release drops the 2026-09-14 family names and the
2026-09-16 nouns together, after the protocols stored on the bench have
been migrated.

### Changes that alter results without raising

- **Twenty-eight operations are renamed; every node fingerprint of
  theirs changes.** The fingerprint hashes the stored identity, so a
  protocol that names any of these starts its cached and resumed work
  over on the next run. Numbers do not change; provenance `operation`
  fields do.

### Other

- **Operations are verbs.** An operation's leaf is now an imperative
  and a kind's a noun, so the two vocabularies never meet on a name:
  `logits/read` emits a `logits/decision`, `records/tabulate` a
  `records/table`, `geometry/compare` a `geometry/similarity`. Before
  this, fifteen operations carried the name of the kind they emitted and
  the documentation site listed `activations/attention` twice. A test
  holds the rule (no operation name is a kind name). The renames, old →
  new — the kinds keep every noun:

  | records | text | eval |
  |---|---|---|
  | `template` → `fill` | `conversation` → `converse` | `expectation` → `expect` |
  | `delta` → `subtract` | `stats` → `measure` | `metric` → `score` |
  | `stats` → `summarize` | | `suite` → `benchmark` |
  | `top-k` → `rank` | | |
  | `histogram` → `bin` | | |
  | `table` → `tabulate` | | |
  | `sum` → `total` | | |
  | `chart` → `plot` | | |

  | logits | activations | geometry |
  |---|---|---|
  | `decision` → `read` | `vectors` → `capture` | `similarity` → `compare` |
  | `funnel` → `read-layers` | `divergence` → `contrast` | `mst` → `span` |
  | `lens` → `scan` | `attention` → `capture-attention` | |
  | `attribution` → `attribute` | | |

  | intervene | direction |
  |---|---|
  | `layers` → `ablate-layers` | `from-vectors` → `fit` |
  | `heads` → `ablate-heads` | `from-pca` → `decompose` |
  | `trace` → `patch` | `vocab` → `unembed` |

  `sum`, `chart` and `trace` were verbs already but the names of the
  kinds they emit; the other twenty-five were verbs and stay
  (`generate`, `select`, `train`, `steer`, `capture`, `project`, …).
  The earlier alias table's targets point at the new names, so
  `decision-read` resolves to `logits/read` in one step.
- **The rollout's per-node top-50 is a partition, not a full sort.**
  Each expansion node of `logits/read`'s rollout sorted the whole
  262k-token distribution to take its fifty most probable children —
  about 20 ms a node, a minute over the 3,136 nodes of a 312-condition
  matrix. It now partitions and sorts the fifty (ties by token id), so
  the children, their order and every number are the same.

## 0.79.0 — 2026-09-15

### Changes that raise

- **`template` is gone from every op; one rendering.** Every model op
  renders its records through `distill.render`: a condition — `user`,
  optional `system`, optional `prefill` — goes through the model's chat
  template as one user turn with the assistant's turn begun by the
  prefill, exactly as `logits/decision` and `text/generate` always
  rendered it; a record carrying only `text` or `prompt`, or saying
  `template: false`, is tokenized raw; `template: "chat"` on such a
  record renders it as the user turn (the retired `template: "raw"` on a
  record reads as `false`). A protocol that still passes the `template`
  param is refused by name. Every interp op can now read at a decision
  point inside an assistant turn; a test holds an ablation sweep's
  baseline to the decision read's number for one prefilled condition.
- **One `tracked` param.** `target` (`intervene/layers`, `heads`,
  `trace`, `logits/lens`, `logits/attribution`), `track`/`tracks`
  (`intervene/apply`, `intervene/steer`) and `outcomes`
  (`logits/decision`, `intervene/apply`) are gone; `tracked: {name:
  token}` names every token an op reports on, the first entry being the
  target a sweep's Δ log p is taken on (the model's own top-1 when none
  is named), and `logits/attribution` decomposes the difference of the
  first two. A record's own `tracked` still takes precedence, and a
  record written earlier with `target`, `outcomes`, `tracks`, `track` or
  `contrast` is still read.
- **One position grammar.** `"last"`, `"all"`, a list of indices,
  `{"tokens": …}`, `{"range": [a, b]}`, `{"after": n}`, `"subject"`,
  `"generated"` — accepted wherever a position is chosen (`positions.py`
  resolves them all). `position` on `activations/vectors`, `intervene/
  steer` and a capture readout takes the same selector and must resolve
  to one position; `"final"` reads as `"last"`. **One pooling clause**:
  `pool: {"reduce": "mean" | "max", "over": <selector>}` replaces
  `pool`/`pool_k`/`pool_skip` on `activations/vectors` (`first_k` after
  `skip` is `{"range": [skip, skip + k]}`, `last_k` is `{"range": [-k,
  null]}`) and `steps` + `reduce` on `trajectory/capture`, where `over`
  counts the trajectory's steps. The retired string form is refused
  with the new one named.
- **One point vocabulary** (`points.py`): `resid_pre`, `resid_post`,
  `attn_out`, `mlp_out`, `gate_out`, `attn.q`, …, `embed`,
  `final_norm`, `logits`. `point: post | pre` on `activations/vectors`,
  `activations/divergence`, `trajectory/capture` and `direction/*`
  becomes `resid_post | resid_pre` (the short forms are still read);
  `intervene/layers`' `component: block | attention | mlp | gate` becomes
  `point`, the sub-layer output(s) zeroed — `attn_out`, `mlp_out`,
  `gate_out`, one or several, both by default (the whole layer, on the
  path that always computed it). `space.point` uses the same names.
- **Metrics live on kinds; two geometry ops stand over any of them.** A
  kind declares how its items compare the way it declares how they are
  drawn: `activations/vector` (and so directions and trajectory points)
  by `cosine` (option `center`), `euclidean` or `dot`; `logits/
  distribution` (and so decision reads, funnels, readouts) by
  `jensen-shannon`, `hellinger`, `total-variation` or `kl`; every record
  kind by `hamming` over `coords`. **`geometry/similarity`** takes any
  such collection on one `items` port (was `vectors`, vectors only) and
  emits the pairwise matrix per group under the named `metric` — the
  kind's first by default — with `metric`, `metric_kind`, `symmetric`,
  `options` and `over` (the kind compared) on its header; `center` is
  now `options: {"center": true}` and is refused as a param.
  **`geometry/mst`** reads only a `geometry/similarity` collection on
  its `similarity` port (was `matrix`; the `vectors` port and `center`
  param are gone) and builds the tree under whatever metric that
  carried, refusing an asymmetric one by name. **`direction/similarity` is retired**: a union of directions
  is a vector collection (`records/union` wraps a single direction as a
  collection of one, named by its port), and `geometry/similarity`
  compares them; the retired name resolves to the new op with a
  warning until 0.80.0, its `a`/`b` ports refused by name.
  `interp.vector_similarity`, `directions.similarity`,
  `similarity_matrix` and `block_similarity` are gone.

### Changes that alter results without raising

- **A chat-templated capture renders as a decision read does.** Ops that
  took `template: "chat"` rendered through the tokenizer's own
  `apply_chat_template` on the model's VLM path; they now render through
  `render_chat` (system merged into the user turn, no special tokens
  re-added), the path `logits/decision` and `text/generate` use. Where
  the two templates agreed, nothing changes; where they did not, the
  capture moves to where the decision was read. The result headers
  record `points`/`pool` in place of `component`/`pool_k`/`pool_skip`/
  `steps`/`reduce`, and no longer carry `template`.

### Changes that accept more

- **The records ops take any collection.** `records/select`, `rename`,
  `delta`, `stats`, `sum`, `top-k`, `histogram`, `table` and `chart`
  declared their `records` port as `records/record | records/table`,
  which since typed ports (0.78.0) refused a collection of decision
  reads, vectors, verdicts or tree summaries by kind — though every one
  of those has an id and fields to summarise, and the tree's own
  description says a table reads it. The port is now `collection |
  records/table`: any collection of items, or a table.
- **`activations/vector` and `activations/grid` extend
  `records/record`**, as the lattice always said: a vector or a grid is
  a record with a space or with axes, so a port typed `records/record`
  takes a collection of either. Their fields are unchanged (`id` and
  `coords` were already declared on both); `logits/distribution` stays
  a root of its own, being a summary that is also a field value.
- **The lexicon declares its families and value types.** `Family(name,
  summary, doc)` for the eleven op families and the four platform
  families; `Value(name, summary, doc, fields, grammar)` for `space`,
  `token`, `top`, `tracked`, `coords`, `derivation` and the three
  grammars `position`, `pool`, `point` — the point page is held to
  `points.POINTS` by a test. `lexicon.FAMILIES`/`VALUES`; the docs
  dump carries both; every kind that had only a summary now has a doc.

---

## 0.78.1 — 2026-09-15

### Changes that raise

- **`records/template` reads its port through the one reader.** The
  0.78.0 registry handed the upstream `collection` itself to the
  template loop, which iterated the container's keys and failed with
  `'str' object has no attribute 'get'` on the first protocol that
  wired a `records/cross` into it. Every records block is now tested
  with a collection on its port, as the executor delivers one.
- **`check_inputs` does not refuse a kind it does not know.** A kind
  string outside the registry — an extension's, or one an author wrote
  before kinds were named — is no name to refuse by; the value passes
  as an unkinded one does. And `"records"`, the spelling every
  experiment author put on its prompt objects, resolves to a
  collection of `records/record`, so those objects satisfy a `records`
  port (0.78.0 refused them: "takes a collection of `records/record`,
  but was wired `records`").

### Changes that alter results without raising

- _None._

---

## 0.78.0 — 2026-09-15

### Changes that raise

- **Inputs are typed ports, and a port is not a param.** Every op
  declares its input ports — name, kind, whether a collection, whether
  required — and the executor checks them before the node runs: an
  edge onto a port the op does not have, a required port with nothing
  on it, and a value whose kind does not satisfy the port's (by
  `extends`) are each refused by name. The params that duplicated a
  port are gone from the lexicon: `records`, `documents`, `vectors`,
  `matrix`, `collection`, `collection_path`, `items`, `direction`,
  `directions`, `against`, `a`, `b`, `trajectory`, `results`,
  `expectations`, `conditions`, `anchors`, `adapter`, `cassette`,
  `participants`, `vocabulary`, `project`. A value that used to go
  there goes under the node's `inputs` — a literal, or `{"$fetch":
  …}` — or arrives by edge. A stored protocol that still carries one
  under `params` runs with a `RetiredParam` warning until 0.80.0, then
  refuses; a protocol that carries one of the removed field-name
  params (below) refuses now.
- **The field-name params are gone.** `system_field`, `user_field`,
  `prefill_field`, `messages_field`, `answer_field`,
  `prediction_field`, `reference_field`, `field`, `fields` and
  `pairwise_fields` (on `eval/judge`), `label_field` and
  `label_coord`. Each op reads the fields it names — `system`,
  `user`, `prefill`; `answer`; `prediction`, `reference`; `text`,
  `text_a`, `text_b` — and a record that carries a value under another
  name goes through the new **`records/rename`** first: `fields: {old:
  new}`, with dotted paths so a measurement can become a coordinate
  (`{"opening": "coords.opening"}`). `value` stays on `records/delta`,
  `records/stats`, `records/sum`, `records/top-k` and
  `records/histogram`: a record has many numeric fields and the port's
  kind cannot say which one the question is about.
- **`text/tokenize` takes its items on the `vocabulary` port** (a word
  list, a weights map, or a bare list of strings) or on `records`; the
  `items` param is gone. It no longer accepts an `adapter`, which
  cannot change a tokenizer.
- **`intervene/apply` reads replacement activations on its `source`
  port only**; the `vectors` spelling of that port is gone.
  `geometry/mst` reads a similarity collection on `matrix` only; the
  `similarity` spelling is gone.

### Changes that alter results without raising

- **A document carries its `coords` on the item**, as every record
  does, beside the copy under `metadata.coords` that older readers
  look for; `text/document` now extends `records/record`, so every op
  that reads records reads a document collection. Stored documents
  are unchanged; new ones carry one more field, which changes their
  content hash and nothing else.
- **A node's fingerprint covers its inline inputs.** What a node
  computes on — wired or given under `inputs` — is part of its
  process identity, so a re-run with a different inline collection
  restarts the node rather than reusing a partial.
- `text/word-list` gains an optional `weights` field: a frequency
  table is a word list whose keys are the words.

---

## 0.77.1 — 2026-09-15

### Changes that raise

- _None._

### Changes that alter results without raising

- **A distribution's `top` ranks tied tokens by id.** Tokens with
  exactly equal log-probability — common at a high-entropy layer read
  through the unembedding, where bf16 logits tie by the hundred — were
  ordered by an unstable sort, so two identical runs could name
  different top tokens where the probabilities were the same. A
  stable sort now breaks ties toward the lower token id, which is also
  what the pre-2b `argmax` did. Found re-running experiment 020: thirty
  of a funnel's 140 top tokens differed between two identical reads.
  Probabilities and entropies were never affected.

## 0.77.0 — 2026-09-15

### Changes that raise

- **A direction is built in a `space`.** `directions.make(vector,
  space, method=…)` replaces `make(vector, layer=…, point=…)`; the
  space is `shapes.space(model=, layer=, point=, d=)`. A caller of the
  old signature gets `TypeError`.
- **The grouping ops take an `axis`, not a `label`.**
  `direction/from-vectors` and `intervene/steer` read the items' value on
  the `axis` coordinate (`direction: {axis, positive, negative}`);
  `direction/from-pca` takes `axis` and `value`; `geometry/similarity`,
  `geometry/mst` and `trajectory/aggregate` group on it. `axis` defaults
  to `label`, which also reads the retired `label` field, so older
  vector collections group as they did. `records/union` no longer
  invents labels: the port is the value on the `batch_axis` coordinate,
  and a direction across a union names `axis: "batch"`.
- **`eval/expectation` emits a collection of `eval/verdict`**, `pass` a
  boolean (null with a `note` when unjudgeable) and the pass rate in the
  header's `summary`; the `ALL` row is gone, and so is the table.
- `intervene/layers` no longer carries a record's baseline as an item
  with `layer: null`; baselines are the header's `conditions`
  (`{id, target, baseline_logp}`), as `intervene/heads` already did.
- `trajectory/capture` with `project`, and `trajectory/project`, emit a
  collection of `activations/coordinate`, not points without vectors;
  `trajectory/aggregate` reads either.
- The op aliases retired in 0.75.0 were to be refused in this release.
  They still resolve (with the warning) until 0.80.0: the protocols
  stored on the bench spell the old names and their migration is
  scheduled work, not a side effect.

### Changes that alter results without raising

- **Every vector item carries its `space`** — `{model, layer, point,
  head, d}` — and the record's `coords`; `layer`, `head` and `label` are
  no longer fields of the item. `activations/vectors`, `direction/*`,
  `trajectory/*`, `intervene/apply`'s captures and `records/union` all
  build through `shapes.vector`; `same_space` is one check. The
  captured token rides along as `token: {id, text}`. Bytes change;
  numbers do not.
- **Every next-token summary is a `logits/distribution`**:
  `entropy_bits`, `top` (ranked `{token, p, logp}`, `top_k` of them) and
  `tracked` (name → `{token, p, logp}` for the tokens asked about).
  `logits/decision` (was `top_tokens` and `outcome_mass`),
  `intervene/apply` and `intervene/steer` (were `top`/`track_logp`/
  `tracks`/`outcome_mass`; steer's alpha is now `factor` and the sweep
  the header's `sweep`), `logits/funnel` (one item per record per layer
  keyed `(id, layer)`, was one document with `metadata.layers` of
  `top1`), `direction/vocab` (two distributions) and
  `trajectory/capture`'s `vocab`. Readers of older reads go through
  `shapes.distribution_of`.
- **Every scalar field over model axes is an `activations/grid`**:
  `axes` and `measures` indexed in axis order. `logits/lens`
  (`logprob`, `rank`), `logits/attribution` (`contribution`),
  `activations/divergence` (`divergence`), `activations/attention`
  (`weight` over `[layer, head, query, key]`), `intervene/trace`
  (`recovery`; `value_a`/`value_b` replace `p_target_clean`/`corrupt`)
  and `intervene/heads` (`mean_delta`). Tokens are `token: {id, text}`
  where a target is named.
- `records/pair` is read by `a`/`b`; `clean`/`corrupt` are still read.
- `intervene/apply`'s capture readout emits `captures` as a collection
  of `activations/vector` (one per hook point, each in its space), and a
  capture readout is accepted on another intervention's `source` port.

## 0.76.2 — 2026-09-14

### Changes that raise

- **`text/generate` and `logits/funnel` refuse an empty record set**, as
  `logits/decision` does since 0.76.1: an empty collection is never
  what a protocol meant. `adapter/train` already refused.
- **Every executor block reads its records through `lexicon.items_of`.**
  `text/generate`, `adapter/train` (records and anchors) and
  `logits/funnel` still took a fetched object's list from `conditions`
  or `records` by name, so a proper `collection` (list under `items`)
  fed to them by param read as empty — `adapter/train` then raised "no
  non-empty training groups". Found by experiment 002's second re-run,
  the first with its battery written as a proper collection. So did
  `text/conversation`'s participants.

### Changes that alter results without raising

- _None._

## 0.76.1 — 2026-09-14

### Changes that raise

- **`logits/decision` refuses an empty battery.** A read over zero
  conditions raised nothing and emitted an empty collection, which a
  reader takes for a finding. It now fails with `logits/decision: no
  conditions to read`. Found the same evening by experiment 002's
  re-run: its battery object had been written by hand as a `collection`
  whose list sat under `records`, and 0.76.0 read `items` only.

### Changes that alter results without raising

- `lexicon.items_of` reads a `collection` whose list was written under
  an older field name (`records`, `conditions`, `rows`), as it already
  did for unkinded objects. A protocol that fed such an object to a
  block and got an empty result in 0.76.0 gets its items in 0.76.1.

## 0.76.0 — 2026-09-14

### Changes that raise

- **Every plural result is now the one `collection` container.** A
  result that used to be its own plural kind (`residual_vectors`,
  `decision_read`, `record_set`, `document_collection`, `similarity_matrix`,
  `mst_summary`, `trajectory`, `patch_trace`, and the rest) is now
  `{"kind": "collection", "item_kind": "<family>/<kind>", "key": […],
  "items": […], …header}`. The list field is always `items`; `rows`,
  `records`, `conditions`, `pairs`, `layers` and `values` as list fields
  are gone from emitted objects. A reader indexing one of those names
  raises `KeyError`. Every block reads the old spellings through
  `lexicon.items_of` and `lexicon.item_kind_of`, so stored objects from
  earlier releases still feed a protocol.
- **Kind names are two-level and bare.** `metric_table` is
  `records/table`, `direction` is `direction/vector`, `adapter` is
  `adapter/lora`, `checkpoint_manifest` is `adapter/checkpoint`,
  `model_pointer` is `model/pointer`, `hf_push` is `adapter/push`,
  `tokenizer_stats` is `text/tokenization`, `pipeline_result` is
  `run/result`, `fs_snapshot` is `sandbox/snapshot`, `completion` is
  `provider/completion`, `provider_cassette` is `provider/cassette`, and
  every item kind path (`~canonical/kinds/text`, `…/transcript`,
  `…/lens-trajectory/2`) is the bare `text/document`, `text/transcript`,
  `logits/funnel`. A reader comparing against a retired string sees no
  match; `lexicon.resolve_kind` maps every retired string to its kind.
  The full table is `lexicon.KIND_ALIASES` and the appendix of
  `docs/LEXICON.md`.
- **Enum parameters are spelled `type`, not `kind`.** A measure, an
  expectation, a sampled factor, a judge scale and an intervene readout
  say `{"type": "lexical"}`, `{"type": "uniform"}`, `{"type": "noise"}`,
  `{"type": "numeric"}`, `{"type": "decision"}`. `kind` is still read
  for this release; it names a kind of object, and the ops' docs no
  longer use it for anything else.
- `geometry/mst` no longer carries a flat `rows` duplicate of its
  per-group statistics: the items are the rows, and `records/table`
  reads them directly.

### Changes that alter results without raising

- **A stored collection's items are in key order.** The executor sorts
  every collection by its kind's key (`id`, or `id, layer, head`, …)
  before hashing and emitting, so the same items in any order are the
  same bytes. An item's position is no longer the order the block
  produced it in; read by key, never by index. Item seeds were already
  derived from keys, so values are unchanged.
- `records/union` of vector collections carries `layers` (the union)
  and `segments` in the header once, instead of once per input.
- `trajectory/capture` and `trajectory/project` mark a projected
  trajectory with `projected: true` in the header rather than a
  different kind; `trajectory/aggregate` emits `trajectory/summary`
  for both vector and coordinate aggregates.

## 0.75.1 — 2026-09-14

### Changes that raise

- _None._

### Changes that alter results without raising

- **`logits/funnel` given its prompts by the `records` param now runs
  over them.** It was the one model block that read `records` only from
  an edge, so a protocol that fetched its prompts by param (`"records":
  {"$fetch": …}`, as the common-parameter page says any block accepts)
  produced an empty collection with no error. Found re-authoring
  experiment 020. A protocol that hit this had no items; it now has
  them.

### Other

- _None._

---

## 0.75.0 — 2026-09-14

### Changes that raise

- **A protocol node whose `block` names nothing refuses by name**, as
  before; what changed is what "names something" means. An op is now
  spelled bare — `records/select` — and every pre-rename spelling
  (`~canonical/ops/decision-read/1`, `grid`, `finetune/lora`) resolves
  through `lexicon.ALIASES` with a `RetiredOpName` warning that names
  the replacement. The aliases are removed in 0.77.0, after which those
  spellings raise. `docs/LEXICON.md` in the meta repo is the rule.

### Changes that alter results without raising

- **Every node fingerprint changes.** The fingerprint hashes the op's
  stored identity, `~canonical/ops/<family>/<op>`, and every op has a
  new family path (the appendix of `docs/LEXICON.md` is the full
  mapping; `direction/*`, `trajectory/*`, `eval/expectation`,
  `eval/suite`, `text/stats` and `tools/calc` are unchanged). Cached
  and resumed work for every existing protocol starts over on the next
  run. Numbers do not change; bytes and provenance `operation` fields
  do.

### Other

- **Fifty-three ops in eleven families** (fifty-four less `grid`), each
  family with at least two members and named for what its ops read or produce: `records/`,
  `text/`, `eval/`, `logits/`, `activations/`, `geometry/`,
  `intervene/`, `direction/`, `trajectory/`, `adapter/`, `tools/`. The
  bare names and the eleven single-child namespaces are gone; `grid`
  is an alias of `records/cross` rather than a registered op;
  `intervene` is `intervene/apply` with its special cases beside it.
  `direction/similarity` keeps its name until 000504 folds it into
  `geometry/similarity`.
- **No version segment.** The trailing `/1` was a path segment that
  happened to be a digit; it is accepted with the same warning as an
  alias and dropped. `@n` is the reserved form for when a contract must
  break.
- **One resolver.** `lexicon.resolve` is called once per node, before
  `check_params` and the fingerprint; `resume_level`, `item_resumable`,
  `algebra` and `check_params` accept any spelling. Every table the
  executor consults is keyed by the bare name, and
  `tests/test_op_names.py` proves it.
- **A source checkout labels itself with its own pyproject version.**
  Under an editable install `__version__` was `<dist metadata>+src.<digest>`,
  and the metadata is whatever number the tree had when `pip install -e`
  last ran — the docs site went out stamped "0.60.0+src…" from a tree at
  0.74.0. The number beside the digest now comes from `pyproject.toml`.

---

## 0.74.0 — 2026-09-13

### Changes that raise

- _None._ `check_params` refuses exactly what it refused in 0.73.0: the
  set of accepted names per op is unchanged, and `test_block_params`
  proves it against the code reads as before.

### Changes that alter results without raising

- _None._

### Other

- **The operation lexicon: `mechbench_compute.lexicon`.** Every canonical
  op is now declared once, as an `Op` with a summary, a description,
  what it takes by edge, what it emits, a runnable example, and every
  parameter as a `Param` with a type, a default (or REQUIRED) and a
  sentence on its effect. `block_params.ACCEPTED` and `COMMON` are
  derived from it; the table that lived there is gone.

  Why a declaration rather than documentation: the docs site was
  rendering parameter NAMES from `ACCEPTED` and a first sentence from
  each docstring, and the result was a page reading "The declarative
  points × operations grammar with a decision or capture readout" over a
  bare list of eight names with no types, defaults or effects. Nobody
  could use the platform from it. Putting the description beside the
  name means the existing bidirectional gate covers it: a documented
  parameter the block does not read fails `test_block_params`, and so
  does one it reads that nobody documented. A docs-side description
  file would have drifted within a week.

  `tests/test_lexicon.py` (220 tests) gates the rest: nothing blank,
  every example accepted by `check_params`, summaries one sentence, and
  no house idioms — a bare task number, "step 07", an epic — anywhere in
  the published text. The docs site's own idiom-stripping and
  "still-internal" report are deleted; the text is authored for a
  stranger at the source.

  Authored against the code, not from memory: each entry was written
  after reading the block, so the defaults are the ones the code uses
  (`intervene.top_k` is 5, `direction/vocab.top_k` is 10; `generate`'s
  temperature is 0.9 and `chat`'s is the provider's unless local).

---

## 0.73.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **Eleven operations gained a first sentence that means something to a
  stranger.** No behaviour changes; docstrings only. The documentation
  site generates each operation's summary from its docstring, so the
  first sentence is now published — and eleven of them read as lab notes:
  `ablate/heads` was "step 07's head sweep", `patch/trace` was "causal
  tracing (step 09)", `direction/add`, `average` and `normalize` had none
  at all. Each now opens with a sentence that stands on its own and keeps
  its in-house detail after it, which is the shape a docstring wants
  anyway: the reader who needs the experiment-step reference is the
  second reader, not the first.

---

## 0.72.0 — 2026-09-13

### Changes that raise

- **All 54 canonical ops now declare their params, so all 54 refuse one
  they do not read** (task 000478). 000438 introduced the check and made
  declaration opt-in — "listing all forty at once would be a refactor
  with no failing test behind it" — which left 46 of 54 ops accepting
  anything and silently ignoring what they did not read. That is the
  same failure 000438 was filed for, standing open everywhere it had not
  been found.

  **A protocol that sets a param one of those 46 ops does not read now
  raises**, naming the param and the block. That is the point, and it is
  the kind of raise worth taking: the alternative is the original bug,
  where six variety jobs asked for `center: true`, all six succeeded, and
  all six were uncentered. Every one of the 46 real protocol nodes across
  the experiment corpus was checked against the new table and accepted,
  so nothing in the existing body of work is refused — but a protocol
  carrying a param that never did anything will now say so.

  One was found that way: experiment 024 set `tool_family: "gemma"` on a
  chat node. Chat has no such param — the tool dialect is DETECTED from
  the model's own chat template (`dialects.dialect_for`), never declared
  — so the Gemma parser was in force regardless and 024's numbers are
  unaffected. The line only misled its reader, which is precisely what an
  ignored param does.

- **`require_resume` joins `COMMON`.** It is authored in a node's params
  and read by the EXECUTOR (epic 000320), never by a block, so it is
  wiring rather than operation. It is not underscore-prefixed like the
  executor's own injections because a protocol writes it.

### Changes that alter results without raising

- **`vectors/mst` no longer accepts `similarity`.** It was declared and
  never read: `similarity` is an input PORT, taken from `inputs`, so a
  protocol passing it as a param was accepted and ignored — the 000438
  bug inside the 000438 fix. No stored result changes; a protocol that
  did this was already not getting what it asked for.

---

## 0.71.0 — 2026-09-13

### Changes that raise

- **`bench.emit` refuses a body over 64 MiB before sending it** (task
  000484), with a `BenchError` naming the size and the limit. The API
  enforces the same ceiling with a 413 (`code: BODY_TOO_LARGE`,
  `limitBytes`, `receivedBytes`); this is the version that says so in one
  line rather than after five retries of a minute each. `MAX_OBJECT_BYTES`
  mirrors `mechbench-api/src/lib/body_limit.ts` and must move with it.
  Legitimate results today are under 1 MB; a body this large usually
  means a record is carrying bytes or a live object it should not (which
  is exactly what 000488 was).

### Changes that alter results without raising

- _None._

---

## 0.70.0 — 2026-09-13

### Changes that raise

- **An un-encodable param or result now raises, by name, instead of
  being hidden** (000488 follow-up). Two places used to swallow it.
  `resume.node_fingerprint` fell back to `repr()` when canonical
  encoding failed; it now raises `TypeError` naming the offending param.
  And the executor hashes a node's result BEFORE emitting it, so a result
  carrying a live object fails locally as `CBOREncodeError` rather than
  surfacing as a network error after the upload is refused. Both were
  exactly what kept 000488 invisible: the fingerprint quietly hashed an
  18 MB Python repr of the adapter bytes, and the emit path turned an
  un-serializable result into "write operation timed out".

### Changes that alter results without raising

- **Node fingerprints for adapted-model nodes change.** The fingerprint
  is now over params in their WIRE form — `{base, adapters}` — where it
  was previously over `repr()` of the resolved object (adapter bytes
  included) once encoding failed. Two consequences: an adapted node's
  fingerprint no longer depends on the fetched payload, only on what the
  run declared, which is what the resume contract always claimed; and a
  spool written by an earlier version for an adapted node will not match
  and restarts that node. No stored result changes value. Base-model
  nodes are unaffected: their params were always encodable.
- `hf/push-adapter` writes the base model ID into the model card rather
  than the resolved ref, which for an adapted `$model` was its repr.

---

## 0.69.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **A generated item no longer embeds the resolved model object** (task
  000488). For a model bound with adapters, `metadata.model` and
  `trace.generation_spans[].model` now record the wire form —
  `{"base": {...}, "adapters": [{"bench": ...}]}` — and `trace.tokenizer`
  records the base model id. Before, all three held the resolved
  `ModelRef`: the first two as a dump of the object, the third as its
  `str()`. That object carries `adapter_payloads`, the fetched safetensors
  bytes, so every story shipped the entire adapter three times — about
  32 MB per item against the 4.5 KB a base-model item weighs. A 20-story
  result was a 640 MB request body; experiment 014's 200-story arm would
  have been ~6.4 GB. The API process (2 GB) was OOM-killed on receipt
  every time, which is what its "write operation timed out" and 502
  responses were.

  Items from a BASE model (a bare HF id) are unchanged byte for byte.
  Items from an adapted model change shape in those three fields, and
  any resumable-node fingerprint or content hash over such an item
  changes with them — which is moot, since no adapted item was ever
  successfully stored. The item's content hash also becomes computable
  at all: `resume.content_hash` canonical-encodes the result directly and
  raised `CBOREncodeError` on the object, which is how the bug was found
  running the block outside the runner.

---

## 0.68.0 — 2026-09-12

### Changes that raise

- **`bench.BenchTransportError`**, a subclass of `BenchError`, is raised
  when a call never got a verdict: a dead socket, a timeout, or a
  502/503/504/429 that survived every retry (task 000464). Existing
  `except BenchError` handlers are unaffected. A host that wants to tell
  "the API was unreachable" from "the payload was rejected" can now do it
  by class instead of by matching the message.

### Changes that alter results without raising

- **`bench._request` retries what carries no verdict.** Five attempts with
  exponential backoff and full jitter, spanning roughly two minutes — long
  enough to ride out a prod deploy's restart window. Retried: a dead or
  timing-out socket, and 502/503/504/429. **Never retried: any other 4xx
  or 5xx**, because a rejected payload does not become acceptable by being
  sent again, and retrying it turns a clear error into a slow one. Repeats
  are safe by construction: object writes are content-addressed, so a
  second attempt writes identical bytes or no-ops.

  This is a correctness fix with a measured cost. Experiment 014's adapted
  run died twice at node `gen` on a single un-retried PUT — once at 197 of
  200 generated stories, once at 200 of 200 — and lost about 35 minutes of
  generation each time.

  **Correction, the same night:** the reasoning given here for that
  incident was wrong, and the retry does not fix it. The failure was
  deterministic, not a blip — prod accepts 30 MB in six seconds and
  **stalls on 72 MB without answering**, which is why the adapted arm
  always failed where the 0.9 MB base arm always succeeded (task 000484).
  Against that, the retry spends five attempts and five minutes on a
  request that cannot succeed. It still earns its place against the blips
  and deploy windows it was written for, but it is not why 014 now
  finishes. A payload over the limit should be refused locally, which
  000484 covers.

  Nothing about a successful call changes, and no result changes value.
  What changes is that a run which previously ended at the first network
  blip now finishes.

---

## 0.67.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **`bench.cancel(job, reason=…)`** withdraws a job that has not started
  (task 000463, api `POST /jobs/:id/cancel`) — the counterpart of
  `launch`, which until now had no undo. Queued or preparing only;
  idempotent, with `alreadyCancelled` on a repeat; a running job is
  refused by the server and surfaces as a `BenchError`. Needs an API on
  or after the 000463 change.

---

## 0.66.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **`trajectory/capture`'s float cap counts what the node EMITS, not
  what it reads.** It counted records × steps × d_model regardless, so
  it refused exactly the two configurations that exist to stay under
  it: `project` (scalars, no vectors) and `reduce` (one pooled vector
  per record). A 100-story, 150-step trace is 23M floats as vectors and
  15k numbers as coordinates; the first is rightly refused and the
  second was wrongly refused with it. Found by experiment 014's
  recomposition, which failed at its last node after 38 minutes of
  generation.

---

## 0.65.0 — 2026-09-13

### Changes that raise

- **A direction record's `provenance` block is now `derivation`.** The
  old name collided with the Emitted envelope's field: `bench.emit`
  refuses to wrap a payload that already carries `provenance`, so a
  `direction/*` node's result could never be emitted inside a run —
  every protocol with a direction node failed at that node with
  "payload already carries provenance". Found by the 018
  recomposition, the first job to run one. A reader of
  `direction["provenance"]` now gets a KeyError; the block's contents
  (`method`, `sources`, `model`, `labels`, per-method extras) are
  unchanged under the new key, and the intervene wire form follows.

### Changes that alter results without raising

- `tokenize/stats` reads `expect_depth` `""` (or `"none"`) as no gate,
  because a protocol hole must always bind and run bindings are
  strings.

---

## 0.64.0 — 2026-09-13

### Changes that raise

- **An adapter whose deltas name modules this architecture does not
  expose is refused, naming them.** Gemma 4's KV-shared tail (layers
  15..34) has no `v_proj` under mlx-vlm 0.6.15; adapters trained in
  August 2026 carry a `v_proj` delta for every layer. `lora.fuse` used
  to fail on the first such layer with a bare AttributeError; it now
  says which modules, on which layers, and how to proceed. Found by
  the 018 recomposition, which fused eleven such adapters.

### Changes that alter results without raising

- **`adapter_skip_missing: true`** (a model-node param, in COMMON)
  fuses the applicable deltas and puts every skipped module on the
  node's result as `adapter_skipped_modules` — the number is never
  without its caveat. Off by default: nothing changes for an adapter
  that fits, and one that does not still refuses.
- `trajectory/capture` gains `reduce: "mean"` over a `steps` window
  (one pooled vector per record — what an outcome axis is fit on) and
  `project: <direction>` at capture time (the scalar trace, no
  vectors). Both exist because a corpus-scale trajectory (200 stories
  × 160 steps × d_model) is ~80M floats, forty times the object cap;
  the outcome axis and the traces are the two small things it was
  ever for. The direction may arrive on an edge.

---

## 0.63.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **A `union` of `residual_vectors` records is a `residual_vectors`
  record** (task 000368), rows labelled by their port when unlabelled,
  the wrapper carried, `layers` the union. It used to be a `record_set`
  that no vector block could read, so no working protocol depended on
  the old output — but a union node over vector records now emits a
  different kind. Cross-model comparison (base vs adapted, experiment
  018) is a union followed by `direction/from-vectors`.
- **`text/stats` `keep: true`** carries the whole item (text, trace,
  metadata) on an annotated row, so a capture downstream can replay
  the story it was labelled on. Off by default; rows are unchanged
  without it.
- **`direction/similarity` over many ports** returns the pairwise
  cosine matrix (`direction_similarity_matrix`: names, cosines, norms,
  sorted pairs). `a`+`b` still return the single cosine.
- `trajectory/aggregate` `as: "vectors"` emits string labels, so a
  text/stats hit (the integer 1/0) meets `from-vectors`' string
  labels.

---

## 0.62.0 — 2026-09-13

### Changes that raise

- **`select`'s `where` now also matches top-level record fields.** A key
  that is not a coord is read from the record itself, so a field that
  `text/stats` `annotate` wrote (a pattern hit) filters. A protocol whose
  `where` key was absent from coords AND absent from the record still
  matches nothing, as before; one whose key was absent from coords but
  PRESENT on the record now matches where it used to match nothing.
  Loud in the sense that a count changes visibly, not silently — but
  audit any `select` that relied on a non-coord key being ignored.

### Changes that alter results without raising

- _None._ Everything else is additive (lexicon epic 000364):
  - **Trajectories as a kind** (task 000368): `trajectory/capture`
    reads the residual stream along one axis — a position across
    layers (the funnel), or a layer across positions along a sequence
    (the trace) — REPLAYING a trace-fidelity record's exact
    `token_ids` from where generation began, rather than
    re-tokenizing its text. `trajectory/project` (scalar coordinate
    along a direction), `trajectory/compare` (per-step cosine, angle,
    norm ratio, the divergence step) and `trajectory/aggregate` (mean
    trajectory + spread per step; a windowed mean per group; or a
    windowed mean emitted as `residual_vectors` so
    `direction/from-vectors` reads it unchanged).
  - **`pool: "first_k"`** on `residuals/vectors`: the windowed read
    (`pool_skip` in, `pool_k` wide) — a story's opening after its
    envelope.
  - **Tokenizer diagnostics** (task 000377): `tokenize/stats` measures
    items — or a `target_map` vocabulary, or records — as continuations
    of a prefix: the depth inventory (histogram, mean, max), the
    single-token fraction, tokens per word, Unicode script
    composition, and the naturalism gate as a pass/fail readout naming
    its violators.
  - **`bench.create_protocol`**: registering a protocol joins
    `launch`/`watch`/`result` in the library; the author half of an
    experiment no longer carries its own `api()`.

---

## 0.61.0 — 2026-09-13

### Changes that raise

- **`bench.fetch` returns the PAYLOAD, not the Emitted envelope** (task
  000450). A reader that indexed the envelope by hand —
  `fetch(x)["payload"]` — now raises `KeyError`; the payload is what
  `fetch(x)` returns, and `fetch_envelope(x)` returns the full envelope
  with provenance for the caller that wants lineage. The idiom every
  experiment reader carried, `(lambda o: o.get("payload", o))(fetch(...))`,
  is now redundant. Internal callers that normalized with
  `o.get("payload", o)` keep working — the unwrap only strips a mapping
  carrying BOTH `payload` and `provenance`, so a payload or a typed record
  is untouched — the loud break is only for a hard index.

### Changes that alter results without raising

- _None._ The rest of 000450 is additive. `bench.launch`, `bench.watch`,
  `bench.results_for`, `bench.result` and `bench.get_job` move the
  launch / watch / find-by-binding / read plumbing that every experiment
  re-derived over httpx into the library — one implementation, which the
  `mechbench run/watch/result` verbs now wrap. And credential resolution
  discovers `~/.mechbench/config.toml` (what `mechbench login` wrote), so
  an experiment script imports neither transport nor key: environment,
  then `configure()`, then that file, with `MECHBENCH_API_KEY` still
  owning the pair when set. No existing result changes.

---

## 0.60.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **A chat item carries its final sandbox workspace** as a browsable
  fs-snapshot on `metadata.sandbox_final` (task 000362). Blobs ride
  INLINE when the whole tree is under 1 MB, so the UI file browser can
  preview file bytes with no blob route; a larger workspace falls back
  to references (shape without content). Mounts are excluded — they are
  read-only inputs, not the session's product. A snapshot is emitted
  as a separate ADDRESSABLE bench object only later, when a large
  workspace makes the per-item inline cost worth the executor plumbing
  (a follow-on).

---

## 0.59.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **The bench-object mount cache.** A read-only mount is
  content-addressed, so its tree is now materialized to the guest
  cache ONCE, under its digest, and every later run — this session or
  another — preopens the same directory. Before, a session's mounted
  stdlib extension was rewritten to disk on every tool call; now the
  first call pays and the rest are free. Concurrent-safe (atomic
  rename, a sibling `.ok` marker outside the mount dir so it stays
  clean).

---

## 0.58.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **Bench-object mounts: the user-extensible stdlib** (task 000453,
  Benji's follow-up). An image may mount a read-only tree at an
  absolute path — `{"path": "/usr/local/lib/python3.13/site-packages",
  "snapshot": {…}}` — and a pure-Python package there is importable by
  the CPython guest. `sandbox.run(..., mounts=[(at, Snapshot)])`
  materializes each read-only as a SEPARATE preopen outside the
  working tree, so a mount is never captured and never writable; the
  working tree at `/` stays the only thing a run changes. An image
  mount that names a bench `object` instead of a `snapshot` stays
  unresolved for the executor to fetch.

---

## 0.57.0 — 2026-09-13

### Changes that raise

- _None._

### Changes that alter results without raising

- **The CPython guest is hosted and pinned** (task 000453). `python`
  now works on any machine: `guests.resolve("cpython")` fetches the
  interpreter (`python.wasm.gz`, pin over the decompressed bytes) and
  its standard library (`stdlib.tar.gz`, pin over the archive),
  verifies both hashes, and unpacks the stdlib into the guest cache as
  a read-only mount. A GitHub release on this repo holds them, with a
  PSF NOTICE. Rebuilt with `-ffile-prefix-map` so the wasm carries no
  build path.
- **Guest runtime mounts can be hosted**: `GuestMount` gains `url` +
  `sha256`; `resolve` fetches and unpacks a `.tar.gz` mount on first
  use (tar extraction guarded with `filter="data"`), reusing the
  unpacked tree by hash. A local build still fills the mount host
  through `install_local(..., mounts=…)`.

---

## 0.56.0 — 2026-09-12

### Changes that raise

- _None._

### Changes that alter results without raising

- **The CPython guest and `python` as a tool** (task 000453). A
  `SandboxSession` now offers `python` (opt-in in the image's tools),
  which runs a `-c` snippet or a script over the SAME snapshot chain
  as the shell tools — a script `bash` writes, `python` runs, and back.
  CPython 3.13.3 compiled to wasip1 (`guests/cpython/build.sh`); the
  clock/RNG/exit lessons from mbshell carry over unchanged, and under
  strict mode `hash()` and seeded `random` are stable.
- **Guests can carry runtime mounts and env** (`guests.GuestMount`,
  `Guest.env`). The CPython guest's standard library is a read-only
  mount at `/usr/local/lib/python3.13` (`fs_mutable=False`, so a guest
  cannot corrupt the shared runtime), found via `PYTHONHOME`.
  `sandbox.run` preopens a resolved guest's mounts and applies its env;
  `guests.resolve(name)` returns `(wasm, mounts, env)`. A guest whose
  reproducible hash is not yet recorded is pinned with an empty
  `sha256` and `install_local` accepts any build for it.
- The CPython guest is declared but **not hosted or hash-pinned yet**:
  the interpreter is ~28 MB and the trimmed stdlib ~10 MB, and the
  build's cross-machine reproducibility is unverified — so `python`
  works only where `guests/cpython/build.sh` has run. Hosting and the
  pin are a follow-up.

---

## 0.55.0 — 2026-09-12

### Changes that raise

- _None._

### Changes that alter results without raising

- **The sandbox catalog** (task 000361): `mechbench_compute.sandbox_kinds`
  pins the shapes the composer (000341) and the UI trace/browser
  (000362) consume — `FS_SNAPSHOT_SCHEMA`, `SANDBOX_IMAGE_SCHEMA`,
  `SANDBOX_TOOL_CALL_SCHEMA`, all tested against the ACTUAL wire output
  of `Snapshot.to_wire`, `SandboxImage` and `SandboxCall` so a schema
  cannot drift from the code. `sandbox_tool_catalog()` gives the tool
  picker its name/description/schema list; `default_image_wire()` gives
  it a ready-to-edit image.
- **`~canonical/kinds/fs-snapshot` is registered** (a `KindManifest` in
  `platform_kinds`): a browsable tree, rendered as a file table
  (`rows=entries`). The image and the tool-call are contracts, not
  renderable kinds — an image is composer config, a tool-call is a
  record inside a transcript — so they are schemas, not manifests. A
  real tree renderer is 000418's job; `table` is the honest default.

---

## 0.54.0 — 2026-09-12

### Changes that raise

- _None._

### Changes that alter results without raising

- **The sandbox tool provider** (task 000360): a chat or conversation
  node may declare a `sandbox` image, and the model is offered tools
  that drive a snapshot chain. `SandboxSession` holds the current
  snapshot; `bash` (mbshell's `sh -c`), `find` and `grep` run in the
  guest and advance it; `read_file`, `write_file` and `list` operate
  on the snapshot directly with no guest. Every call is recorded on
  the item's `metadata.sandbox` as `{tool, argv, exit_code, limit,
  snapshot_in, snapshot_out, changed, stdout, stderr}` — the
  filesystem's history call by call. Tools are ours, thin and few
  (the MCP decision stands); the image (`SandboxImage`) is the
  standard-library base users augment — tool allowlist, limits,
  strict mode, read-only mounts, starting tree.
- A tool handler may now be `{"sandbox": "<method>"}`, dispatched
  against a session bound to the `Toolbox` (stateful, so it does not
  ride in the handler dict, which is copied into every provenance
  record). `toolbox_from(..., session=…)` binds it. Existing
  `{"block"}` and `{"protocol"}` handlers are unchanged.
- The `chat` block accepts a `sandbox` param. A node without one is
  unchanged — same items, no `sandbox` key. Resume replays a spooled
  item wholesale, so a recorded session is byte-identical without
  special handling.

---

## 0.53.0 — 2026-09-12

### Changes that raise

- _None._

### Changes that alter results without raising

- **mbshell is hosted.** `guests.ensure("mbshell")` fetches it on
  first use from a GitHub release on this repo
  (`mbshell-5090c2c1646c`), gzipped, and verifies the decompressed
  bytes against the pin; a local build is no longer needed to run
  the sandbox. `NOTICE` ships beside the binary with every bundled
  license. Benji's call on the license question: go-busybox declares
  MIT in its README, and a declaration is a license — the missing
  file is upstream issue #3, not a blocker.
- `_fetch` decompresses a `.gz` URL on the way down; the pin is
  always the hash of what runs. It trusts certifi's CA bundle, as
  `bench.py` and `providers/http.py` already do — the python.org
  framework build on macOS has no system bundle, and the first real
  fetch failed with CERTIFICATE_VERIFY_FAILED.

---

## 0.52.1 — 2026-09-12

### Changes that raise

- _None._

### Changes that alter results without raising

- **mbshell pin `5090c2c1…`**: built with `-buildvcs=false`. The
  recipe sits inside the compute git repo, and Go stamped the binary
  with that repo's commit and dirty flag, so the 0.52.0 pin
  reproduced on this machine and on no other. CI's first build of the
  guest hashed differently and every guest test refused to run —
  which is the check working. Now the same hash from inside the repo
  and from a copy outside any git repo.
- Cached wasmtime modules and the engine are released at interpreter
  exit, before the FFI is unloaded; their finalizers printed a
  `TypeError` on shutdown.

---

## 0.52.0 — 2026-09-12

### Changes that raise

- _None._

### Changes that alter results without raising

- **The compiled guest is cached.** Compiling the 15 MB standard-Go
  module took 1.1 s on every call; a tool call is otherwise 5 ms.
  One engine per process, one epoch ticker, a store per run, and the
  compiled module kept in memory with a serialized copy in the guest
  cache (10 ms to load, keyed by guest identity and wasmtime version).
  Found by the conformance battery: 456 runs took ten minutes, now 48 s.
- **`limit="stack"`** names the wasm call stack ceiling (wasmtime's
  512 KiB default) that unbounded recursion in the guest hits — it
  was an unclassified trap.
- **Guest conformance battery** (`tests/test_guest_battery.py`):
  every applet against file, empty, binary, large, unicode,
  directory, deep, many-files and missing inputs, plus the no-path
  and network applets, asserting the robustness contract — no trap,
  no panic, bounded time, read-only means read-only. 488 runs pass.
  A new applet upstream fails the coverage test until templated or
  excluded with a reason. CI now builds the guest from the pinned
  recipe before the tests, which refuse a build off the pin, so every
  push is also a reproducibility check.

---

## 0.51.0 — 2026-09-12

### Changes that raise

- _None._

### Changes that alter results without raising

- **mbshell is built with standard Go's `wasip1` port, not TinyGo.**
  Every guest failure found by running it traced to TinyGo: no
  `recover` on wasm, a negative read count on directories that bufio
  panics on (`wc -c .` killed the sandbox), and the reflect gap that
  broke `awk`. Same source, one build flag; the binary is 15.4 MB
  (3.8 MB gzipped) against 2.8 MB, fetched once. New pin
  `8105ef5a…`. `awk` works; a directory argument is an ordinary
  non-zero exit.
- **`xargs`, `time` and `timeout` run their command in-process**
  through a seam added to go-busybox (`go-busybox-wasi.patch`:
  `core.RunCommand`), which mbshell points at its applet table.
  `timeout`'s duration is parsed and not enforced — nothing to signal;
  the sandbox wall cap is the only clock. `find` has no `-exec`
  upstream.
- **Waiting is virtual, in both modes.** `poll_oneoff` was denied in
  0.50.0; Go's runtime waits inside its GC path, so any guest that
  grew its heap died. Now every wait completes at once and the guest
  clock jumps forward by the wait (host time plus skipped waits, or
  the strict counter), so `sleep 30` returns in milliseconds and
  `time sleep 5` reports five seconds. `limit="blocked_call"` no
  longer exists.
- **Exit statuses ≥ 126 are carried.** WASI hosts reject `proc_exit`
  outside [0, 126) and drop the number — and 127 is "command not
  found". The guest writes the real status to a second preopen,
  `/.mechbench/exit`, and exits 125; the runtime reads it back.
  Without the side channel the floor, 126, is reported.
- `sh -c CMD NAME ARGS` sets `$0` to NAME, as POSIX says.
- A memory cap below the guest's declared minimum (113 pages here) is
  reported as `limit="memory_mb"` with the minimum named, not raised.
  Go's out-of-memory exit (status 2, message on stderr, memory at
  the cap) is classified as `memory_mb` too.

---

## 0.50.0 — 2026-09-12

### Changes that raise

- **`sandbox.run(snapshot, argv, guest=…)` exists** (task 000359).
  Runs a WASI guest over a content-addressed snapshot with one
  preopened directory, no network, fuel-metered CPU, an epoch wall
  clock, a memory cap and an output cap; every ceiling that trips is
  named in `Result.limit` (`fuel`, `wall_seconds`, `memory_mb`,
  `max_files`, `max_bytes`, `blocked_call`). `poll_oneoff` is always
  denied: a guest blocked in it is beyond epoch interruption (`sleep
  10` under a 1 s cap ran 10,002 ms), and no tool call needs to wait.
- **`guests.ensure(name)` fetches a pinned guest on first use and
  verifies it by hash**; wrong bytes are deleted, never run. A local
  build goes in through `guests.install_local`, which refuses a build
  that differs from the pin unless told `replace=True`.
- **The guest is `mbshell`** (`guests/mbshell/`, buildable with
  `build.sh`): go-busybox's applets (go-busybox `13f3053`, TinyGo
  `wasip1`) behind an in-process POSIX shell, because go-busybox's
  own `sh` is fork/exec and WASI cannot spawn a process — it is
  stubbed upstream, so plain busybox is not pinned at all. mvdan/sh
  v3.12.0 with a four-item WASI patch (`io.Pipe`, `io.Reader` stdin,
  existence-only `access`, deadline only where supported) runs
  pipelines on goroutines and hands every command to the applet
  table. Pipelines, redirects into the snapshot, `cd`, `$(…)`,
  heredocs, `set -e`, `/dev/null` all work; strict runs are
  byte-identical. Applets that exec a command themselves (`xargs`,
  `find -exec`, `timeout`) do not work yet. **Not hosted**: upstream
  claims MIT with no LICENSE file in the tree.
- New dependency: `wasmtime>=48` (8 MB, native wheel).

### Changes that alter results without raising

- **Strict mode virtualizes the clock and RNG rather than denying
  them.** Denied, nothing ran: the TinyGo runtime reads the clock
  before `main`, and CPython seeds its hash from `random_get` before
  the first line. Strict now serves a clock starting at
  2000-01-01T00:00:00Z that advances 1 µs per read, and SHA-256 bytes
  seeded from the snapshot digest and argv. Verified on the one
  witness a shell gives for free: Go map order, which `busybox`
  shuffles every plain run and holds fixed under strict.
- `Snapshot` carries an in-process `blobs` sidecar for content above
  the inline threshold (0.48.0 extension, now used by `materialize`
  when no store is passed). Not part of identity or the wire form.

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
