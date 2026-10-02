# Where an operation lives

Every operation is one file, and the file's path is a function of the
operation's name.

```
intervene/patch
  docs     https://docs.mechbench.ai/ops/intervene/patch/
  source   mechbench_compute/ops/intervene/patch.py
  tests    tests/ops/intervene/test_patch.py
```

A hyphen in a name is an underscore in a path: `logits/read-layers` is
`ops/logits/read_layers.py`. Nothing else is translated. If you know an
operation's name you know where its contract, its code and its tests
are, without searching.

## What an operation's file holds

```python
OP = Op(name="intervene/patch", ...)      # the contract: ports, params, output, prose

def run(ctx, inputs, params):             # what the executor calls
    ...

def patch_trace(model, records, params):  # the mechanism, callable on its own
    ...
```

- **`OP`** is the declaration the documentation site renders and
  `check_params` enforces. Besides the ports, params and prose it says
  what the operation asks of the host (`needs`), what a re-run may reuse
  of it (`resume`), and whether a double run is identical
  (`deterministic`). It is the only place the operation's
  parameters are written down, and it is complete here: a param several
  operations declare the same way, a port they share, a paragraph of
  prose — each is written out in the file rather than imported, so the
  contract reads whole without opening anything else. The only thing
  the file takes for it is the vocabulary a declaration is written in —
  `Op`, `P`, `In`, `Output`, the kinds — from `mechbench_compute.api`
  (docs/PLUGIN_API.md).
  Two operations that say the same thing say it twice, on purpose:
  a declaration is prose for a reader, and the reader has one file
  open.
- **`run(ctx, inputs, params)`** is the one entry point, the same
  signature for every operation. It unpacks `inputs`, asks `ctx` for
  what it needs, and calls the mechanism.
- **The mechanism** is one or more plain functions that take loaded
  things — a model, a list of records — and return the result. Tests
  call these directly.
- **`MONOID`**, when the operation can be computed in chunks: the class
  that reduces them. Its presence is the declaration.
- **`read_resume_level(params, inputs=None)`**, when what a re-run may
  reuse depends on the params (`text/chat` answered by a provider is
  `exchangeable`, by local weights `reproducible`). `resume.resume_level`
  asks it before falling back to `OP.resume`.

An operation's file holds what only that operation uses.

## What operations share

This is about the code operations run. A declaration is not shared: two
operations that declare a param the same way each write it out, because
a contract is prose and its reader has one file open.

A definition two or more operations use is a file of its own, under the
topic it belongs to, named for itself:

```
mechbench_compute/interp/read_last_logp.py   def read_last_logp(logits)  12 lines
mechbench_compute/intervene/spec.py          class Spec                 257 lines
mechbench_compute/directions/coerce_array.py def coerce_array(value)     22 lines
```

So the import line at the top of an operation's file is the path to
open — `from mechbench_compute.interp.read_last_logp import read_last_logp` —
and opening it costs that definition and nothing else: a median twelve
lines. Most of the time the name is enough and the file is never opened.

A constant only one helper uses lives in that helper's file; a constant
several things use lives in `<topic>/constants.py`. A class and the
function that builds it (`Plan` and `plan`) share a file.

Three other arrangements were measured before this one, by what one
agent reading one operation has to take in:

| | median |
|---|---|
| shared code in topic modules, opened whole | 446 lines, 3 files |
| shared code filed by who uses it (`_common.py`) | one file of 1,614 lines |
| helpers copied into each operation's file | 186 lines, 1 file — and the copies drift, and a copied class loses its identity |
| **one helper per file** | **165 lines if it opens none, 195 if it opens every one** |

It works because of a fact about this code rather than a preference:
among the 114 definitions operations share there is not one reference
cycle, so each can be a file with plain imports at its top.

A package's `__init__.py` says what the topic is, and imports back only
the names something still reaches through the package rather than
through the helper's own file — `from mechbench_compute.interp import
read_pair`, a test that patches `judge.chat_mod`. A helper everything
names by its own path is not listed there at all, so the `__init__.py`
is a short and shrinking list rather than a second index of the
package. An operation's file is a leaf: nothing imports back from it.

## What `ctx` offers

`ctx` is `mechbench_compute.ops.Context`: what the executor lends an
operation for the duration of one node. `ops.read_context_uses(mod)`
reads, from the code, which of these an operation's `run` touches; the
lexicon tests hold every operation's `needs` to it, and at run time the
executor builds the context with `Context.for_op`, which lends a member
only to an operation that declared a need for it:

| member | lent to an operation that needs |
|---|---|
| `ctx.model`, `ctx.loaded`, `ctx.evict_model` | `model.forward`, `model.sample` or `model.backward` |
| `ctx.sub`, `ctx.executor` | `executor.sub` |
| `ctx.provider` | `provider.chat` or `provider.embed` |
| `ctx.memo` | `memo` |
| `ctx.materialize` | `objects.read` |
| `ctx.secrets` | `secrets` |

Any other use raises `NeedNotDeclared`. The rest of `ctx` — progress,
resume state, paths, run params — is lent to every operation.

| | |
|---|---|
| `ctx.model(ref)` | the model for a `model` param: `ctx.loaded` when one is lent, otherwise the executor loads it |
| `ctx.loaded` | a model already in hand, which `ctx.model` returns whatever the ref |
| `ctx.sub(target, inputs, params, *, budget, on_item, on_start)` | run one operation (`target` is its name) or a graph (`target` is a mapping in the protocol form) as a child run that shares the resident model and the rate limiter; `budget` is a cap in USD under the run's own budget. A graph answers with its outputs by node |
| `ctx.provider(model_ref)` | a `ProviderClient` for a provider's endpoint, carrying the rate limiter, the run's budget and `ctx.secrets`: `chat(records, params, …)`, `embed(…)` |
| `ctx.memo(key)` | the memo of remote calls a `cache` param names (`None` for none): its `tape`, and `close(out)` to store it and note the hits on `out` |
| `ctx.materialize(label)` | a stored checkpoint, as a local directory |
| `ctx.evict_model()` | drop the resident model, for an operation that changed its weights in place |
| `ctx.on_start`, `ctx.on_item` | progress: one call when the count is known, one per item |
| `ctx.on_token` | streaming: one call per generated token (`text/generate`, `text/chat`) |
| `ctx.on_checkpoint` | for an operation that checkpoints (`adapter/train`) |
| `ctx.resume_items`, `ctx.resume_state` | what an interrupted run already finished |
| `ctx.secrets` | the owner's provider credentials |
| `ctx.input_paths` | the stored label each input arrived from |
| `ctx.run_params` | the run's bound params, which a `$param` in an expression reads |
| `ctx.result_base` | where the run's results land |

Every field has a default, so a test builds one in a line:
`run(Context(loaded=fake), inputs, params)`.

### The executor boundary

An operation reaches the executor only through these members. No file
under `ops/` or `live/` names `ctx.executor.` or `type(ctx.executor)`
(`test_no_op_reaches_into_the_executor`), and every member an
operation's `run` touches is named by a need it declares.

## What is derived, and so is not written down

- **The registry.** `mechbench_compute.registry` walks every module
  under `ops/` and `lexicon/kinds/`. There is no table of operations to
  keep in step with the files. How it is built is the last section.
- **Where it runs** — `OP.requires`, derived from `OP.needs`: a
  `model.*` or `runtime.mlx` need is `local` (`mlx-local` before the
  backend was named apart); otherwise a
  `provider.*` or `network:*` need is `remote`; a `model.*` need with a
  `provider.*` or `network:*` one is `by-model`; none is `pure`. It is
  still in the wire form, for placement; it is no longer written.
- **Whether it can run with no executor at all** — it needs nothing and
  its `run` never reads `ctx`. That is the set a tool handler or a chunked
  reduce may call.
- **Whether the executor fuses an adapter around it** — it needs
  `model.forward` *and* declares an `adapter` input port: there are
  weights to fuse onto, and an adapter may arrive. The port alone does
  not say so — `adapter/measure` and `adapter/publish` take an adapter
  as the thing they operate on. An operation that runs either side
  (`by-model`) with an `adapter` port is fused when its node runs local
  weights rather than a provider's endpoint.
- **Whether it can be computed in chunks, and how** — its file defines
  `MONOID`. That one line is what the reduce machinery, the chunk
  harness and the resume levels all read.

A test that wants to stand in for an operation patches the operation's
own file — `monkeypatch.setattr(fill, "run", flaky)`,
`monkeypatch.setattr(total, "MONOID", Bad)` — because that is where it
runs from.

## Where a kind lives

The same rule as an operation's: one file per kind, its path a function
of its name.

```
logits/distribution   mechbench_compute/lexicon/kinds/logits/distribution.py   KIND = Kind(...)
text/word-list        mechbench_compute/lexicon/kinds/text/word_list.py
collection            mechbench_compute/lexicon/kinds/collection.py
```

The registry walks the tree (last section); `lexicon/kinds/__init__.py`
holds `KINDS` and `BY_KIND`, views over it, and the helpers every reader
of a kind uses
(`ancestry`, `satisfies`, `all_fields`, `resolve_kind`, `item_kind_of`,
`items_of`, `collection`, the retired spellings). The platform's own
kinds — the `run`, `sandbox`, `provider` and `model` families — live in
the same tree and say `platform=True`; there is no second tree for them.
A field fragment several kinds share (`TOP`, `TOKEN`, `DIST`) is a value
in `lexicon/values.py`.

## Two rules the layout depends on

**An operation's file imports cleanly on a machine that cannot run it.**
The lexicon is read by the documentation build, by the API, and by
runners without Apple silicon. `mlx` comes from
`mechbench_compute._mlx`, which raises a clear error on first use rather
than at import.

**No file is over the size budget.** Six hundred lines, held by a gate in
the suite (`tests/test_file_budget.py`). A 3,000-line file is not
something an agent can read to understand one operation, and nothing else
keeps a file small. The files already over the budget when the gate went
in are listed there at the length they had that day: each may shrink,
none may grow, and one that drops under the budget comes off the list —
so the list only ever gets shorter.

## Where the executor lives

The same two rules, applied to the thing that calls an operation's `run`.
`ProtocolExecutor` is composed from one mixin per topic, each a file
under `mechbench_compute/protocol/` named for its topic:

```
pipeline.py       walking the graph, and nothing else
dispatch.py       what answers for a node, and the one hop to it
model.py          loading weights, fusing adapters
remote.py         the nodes a provider answers, run in a wave
memo.py           a node's memo of the remote calls it made
sub.py            a child run that shares the resident model
legacy_kinds.py   the two spec kinds that came before the graph
```

What they share is one definition per file, as `serialize_params` and
`read_tokenizer_id` already were. The class is still one class, and its
methods and their call sites are unchanged: a mixin is how a method keeps
its `self`.

The walk itself is not a mixin's worth of topics but one loop over the
nodes, and each thing it does at a node is a definition in a file named
for it, under the same rule as any other helper: `RunState` (what a run
carries from node to node), `Resolver` (a node's references made
values), `Progress` (what a watcher is told), and `sort_nodes`,
`gather_inputs`, `read_resume_entry`, `restore_node`, `store_result`,
`check_failures`, `build_manifest`. The loop passes the state to each,
so the walk reads as what it is: order the nodes, and for each one
resolve, dispatch, record.

## The registry, and where an operation comes from

`mechbench_compute/registry.py` answers "what runs this block?" for
every caller. `REGISTRY` is built from sources, in order:

- **`CoreSource`** walks `ops/` and `lexicon/kinds/` with the per-file
  walker in `lexicon/walk.py` (`walk_ops`, `walk_kinds`: one file per
  declaration, its path a function of its name). The walk happens once,
  on the first read, and is held by the source.
- **`InstalledSource`** reads the entry points in the group
  `mechbench.extensions`. Each names an `Extension` (`pkg:MANIFEST`),
  whose package has the same `ops/` and `kinds/` layout as core's and is
  walked by the same walker. Its operations register under
  `<owner>/<project>/ops/<family>/<leaf>`, its kinds under
  `<owner>/<project>/kinds/<family>/<leaf>`; a port or output that names
  one of the extension's own kinds by its short name is rewritten to
  that address, and so is a kind's `extends`. An extension is refused at
  load, and its operations answer with why, when its owner is a core
  family name (family names are reserved handles, so a bare name and an
  extension address never collide), when a kind extends anything but a
  core kind or one of its own, when a kind is in a sealed family or has
  no `speak`, or when an operation names a kind nobody declares.

`REGISTRY.resolve(spelling)` returns a `Resolved` — the declaration
(`op`), its `tier` (`core` or `installed`), the `module` whose `run`
executes, its `name` (the key everything else uses) and its `source`
(`core`, or `<owner>/<project>/extensions/<name>@sha256:<digest>`):

| spelling | answers |
|---|---|
| `family/leaf`, `~canonical/ops/family/leaf` | the core operation |
| `<owner>/<project>/ops/<family>/<leaf>` | the installed extension's operation |
| `…@<n>` | the same, only if the installed version is `n` |
| `…@sha256:<digest>` | the same, only if the installed version's digest is that one; otherwise a `KeyError` naming both digests |
| anything else | a `KeyError` whose message is `lexicon.explain_unknown`'s |

A version is never a path segment: `records/filter/2` is refused, and
so is `…/ops/geometry/align/2`. The grammar is `lexicon/address.py`
(`parse`, `parse_op`, `parse_kind`, `parse_mark`, `parse_extension`,
each an `Address` with `.bare`, `.version`, `.pin`, `.is_core`); it is
the same grammar as `readAddress` in mechbench-models.

`refresh()` walks the entry points again after an install and bumps
`generation`; anything memoized by name (`find_standalone`) keys its
cache on it. It answers `{added, dropped, refused}`: the extension
addresses loaded for the first time, those forgotten with the reason
(no entry point provides it any more, or the pin its install recorded
left `installed.json`), and those refused at load with the reason.
`refused` is keyed by the extension's address
(`<owner>/<project>/extensions/<name>`), or by the entry point's name
when the manifest itself could not be read. A forgotten extension's
ops and kinds leave the registry at once; its modules stay imported.
Python cannot unload a module, so an extension whose version changed
after it was loaded (or after it was forgotten) is kept at the loaded
version and `refresh()` raises `RestartRequired`: the runner restarts
its `run` child.

**The digest.** `hash_extension(manifest)` is `sha256:` over the
canonical JSON of the manifest's declaration, `canonicalJson(declarationOf(m))`
in mechbench-models byte for byte, so the API's pin and compute's agree
on the same declaration. The declaration is `PIN_FIELDS` (`owner`,
`project`, `name`, `version`, `tier`, `provides`, `needs`,
`min_compute`, `package`, `links`) of the manifest as models' zod
schema parses it: every default filled (`provides.{ops, kinds, marks,
architectures}`, `needs`, `links`; each op's `inputs`, `params`,
`output`, `resume`, `deterministic`; each param's `doc`; each port's
`doc`, `required`, `many`, `variadic`, `on_missing`; each output's and
`otherwise`'s `collection`; each kind's `doc`, `fields`, `required`,
`extends`, `key`, `header`, `metrics`, `platform`, `version`; each
metric's `doc`), optional keys never written left absent, `owner` and
`project` lowercased. The JSON has keys sorted by UTF-16 code unit, no
whitespace, strings as `JSON.stringify` writes them, numbers in
JavaScript's shortest form (`1` for `1.0`, `1e-7`, `1e+21`). The
declaration the API hashes carries what only the push knows
(`package.sdist`), so an installed extension's digest is the hash its
install recorded in `~/.mechbench/extensions/installed.json`;
`hash_extension` over the package's own `to_dict()` is the fallback for
one installed by hand.

**The view rule.** Everything that read a table now reads the registry,
and the old names are views over it so that their import sites keep
working: `lexicon.OPS` and `kinds.KINDS` are sequences, `lexicon.BY_NAME`
and `kinds.BY_KIND` mappings, each read afresh from `REGISTRY` on every
access, so an extension installed after import is in them.
`ancestry`, `satisfies`, `all_fields` and `resolve_kind` read kinds
through `BY_KIND`; `block_params.accepted(block)` and `ports(block)`
replace the tables that were built at import; `check_graph`, dispatch,
`is_remote` (which reads `"provider.chat" in op.needs`), `fuses_adapter`,
`reduce.algebra` (`MONOID` on `Resolved.module`), `resume.resume_level`
and `Toolbox` ask the registry. `ops.find(name)` returns the module, for
one more release. What is generated for the other repositories —
`lexicon.generated.ts`, `kinds.generated.ts`, the kind manifests — reads
`CORE` only, so an extension on the machine that generates them never
leaks into them.

The lexicon and the operations do not import each other: the registry
imports both, and a view imports the registry only when it is read.

## Where an architecture lives

A model architecture — a checkpoint's config.json `model_type` — is one
file, and its path is its `model_type`:

```
mechbench_compute/architectures/gemma4.py   ARCH = Architecture(model_type="gemma4", ...)
mechbench_compute/architectures/gemma3.py
mechbench_compute/architectures/llama.py
mechbench_compute/architectures/qwen2.py
```

`architectures/__init__.py` walks the package (a leaf starting with `_`
is a shared helper, not an architecture) and refuses a file that
declares no `ARCH` or declares a `model_type` other than its name.
`architectures.ARCHITECTURES` is the walked tuple, `for_type(model_type)`
finds one, `for_model(model)` finds a loaded model's. `support.refusal`,
`support.local_architectures` and `_arch.family_supports` read the
walked set; nothing else lists architectures.

Those are the `mlx` backend's. An architecture is implemented per
backend (docs/CAPABILITY.md §2.6 in the meta repo), and the `torch`
backend's live under its own package, walked the same way:

```
mechbench_compute/torch_backend/architectures/gemma3.py   ARCH = Architecture(model_type="gemma3", backend="torch", loader="transformers", ...)
mechbench_compute/torch_backend/architectures/llama.py
```

What an architecture declares apart from any backend lives where both
read it without importing either: its tool dialect in
`tool_dialects/<model_type>.py` (`DIALECT`, walked by
`dialects.list_dialects`), its adapter key map in `adapter_keys.py`.
`Architecture.backend` names the backend an implementation is for; the
torch package imports MLX nowhere, and nothing outside it imports torch.

The torch backend runs a `transformers` model under an nnsight trace
(`torch_backend/forward.py`): a run with no hook and no capture is the
model's own forward, and a run with any is one trace that reads and
writes only the points named, in the order the forward reaches them,
through the modules (`layers[i].input`/`.output`, the post-attention and
post-feedforward norms, `lm_head`) and through nnsight's source
operations inside the attention (`transpose_2` for `attn.v`,
`apply_rotary_pos_emb_0` for `attn.q` and `attn.k`,
`attention_interface_0` for `attn.per_head_out`, and the eager
attention's `nn_functional_dropout_0` for `attn.weights`). Attention
runs as SDPA unless `attn.weights` is named, when that run is eager.
`torch_backend/tracing.py` loads nnsight with its `.save` mount turned
off, since an MLX imported after that mount aborts.

`support.Architecture` is the interface. Its declarative fields:
`model_type`, `name`, `loader`, `generate`/`score`/`train`,
`layer_points`, `global_points`, `refused_when` (config keys that refuse
the checkpoint), `absent_when` (points a checkpoint lacks when a config
key is 0 or unset: it loads, and a hook there is refused by name),
`config_defaults`, and
`residual_law`, a string the kit evaluates
(`"resid_post[i] == resid_pre[i] + attn_out[i] + mlp_out[i] == resid_pre[i+1]"`,
or the architecture's own; `residual_law_of(arch)` drops an absent
point's term, and `writes_of(arch)` reads the writes it adds, the
`*_out` terms, which `logits/attribute` splits a layer into). Its
callables, which `Model` delegates to
without asking which architecture it holds:

| field | signature |
|---|---|
| `load` | `(model_id, **config) -> (model, processor)` |
| `arch_of` | `(model, model_id=None) -> Arch` |
| `forward` | `(model, ids, *, hooks, capture, arch, kv_cache) -> (logits, ActivationCache)` |
| `lm` · `prompt_cache` | `(model) -> ...` |
| `head_logits` · `project_to_logits` | `(model, hidden) -> logits` |
| `tokenize` | `(model, processor, prompt, *, chat_template) -> ids` |
| `attribution_unembed` | `(model) -> Unembed(norm, project, softcap)` |
| `layer_scalars` | `(model) -> tuple[float, ...]`, each layer's scalar on the whole stream when the residual law has `layer_scalar[i]` (Gemma 4); `None` otherwise |
| `attn_out_norm` | `(model, layer) -> norm`, the norm `o_proj`'s output passes through to become `attn_out` (Gemma's `post_attention_layernorm`); `None` when `attn_out` is `o_proj`'s output (Llama and Qwen 2, whose `post_attention_layernorm` is the MLP's input norm) |
| `head_weights` | `(model, layer, head) -> HeadSpec`, or `NotImplementedError` naming the architecture |
| `dialect` | a `dialects.ToolDialect`, or `None` when the template has no tool protocol |
| `reasoning` | `thinking.Delimiters` the model's tokenizer may declare |
| `adapter_keys` | `lora.AdapterKeys(key_re, containers, peft_re)` |

The forwards share `architectures/_dispatch.py` (`dispatch`, the one
hook-and-capture step, and `run_head`, the final norm and unembedding);
the two loaders' plumbing is `_vlm.py` and `_mlx_lm.py`.

### The kit

`tests/test_architecture_kit.py` runs per backend, over every
architecture the backend implements, with a tiny random model of it
(an architecture without one fails the kit), and over each variant
whose config changes which points exist (`VARIANTS`: `gemma4-31b`).
A backend's tiny models are a `KitBackend` (`tests/kit_backends.py`):
MLX's in `tests/tiny_models.py`, torch's in `tests/tiny_torch_models.py`
(`transformers` models with random weights: Gemma 3 text, Gemma 3 with
its vision tower, Llama; on the CPU, or the GPU when
`MECHBENCH_KIT_DEVICE=cuda`), and a fake second backend's in
`tests/fake_backend.py` (numpy arrays over MLX's forward). Each is
selected by what a runner that has it advertises
(`backends.select`), and skipped by name on a machine that lacks it.
On a Linux machine with an NVIDIA GPU:

```
pip install -e '.[torch,dev]'
MECHBENCH_KIT_DEVICE=cuda python -m pytest tests/test_architecture_kit.py tests/test_torch_backend.py tests/test_backends.py -q
```
The checks read arrays through `mechbench_compute.arrays`, so they
hold for any backend's arrays:

- every declared point is captured, in the shape `points.LAYOUT` gives;
- the global points agree with the stream (`embed` is `resid_pre` of
  layer 0, `final_norm` is the norm of the last `resid_post`, `logits`
  are `head_logits(final_norm)`);
- the head applies the final logit softcap the checkpoint's config
  sets, and declares it (`Unembed.softcap`); a head that applies none
  is the bare projection, and its architecture refuses a config that
  sets one;
- the residual law holds at every layer, to 2^-6 of the largest
  magnitude (bf16 keeps 8 significant bits; the law is summed in
  float64 over values the forward added in bf16), its `layer_scalar`
  read through the declared `layer_scalars`;
- the writes and the scalar the attribution reads are the ones the
  residual law names;
- direct logit attribution with the final norm sums to the true logit,
  by layer and by sublayer; a layer's piece is the sum of its sublayer
  pieces, and without scalars the stream's step at that layer;
- a layer's heads, through the declared `attn_out_norm`, sum to its
  `attn_out`;
- `activations/capture` reads the residual law's points, and its
  vectors at the last position and pooled over the sequence add up by
  the law; it reads `gate_out` where a layer writes one and refuses it by
  name (`POINT_ABSENT`) where none does; its `attn_out` split by head
  sums to its `attn_out`, each head's part being what ablating the head
  removes, up to the post-attention norm's divisor; its attention
  weights sum to 1 over the keys and are causal; its `top` names each
  vector's largest coordinates, their shares of the squared norm and
  the rms without them;
- capturing attention internals leaves the logits alone, and attention
  is causal;
- ablating every head of a layer equals zeroing its `attn_out`;
- a double run is bit-identical;
- `tokenize` round-trips through the tokenizer;
- the dialect parses the tool call its own template rendered
  (`tests/fixtures/chat_templates.json`);
- the adapter keys reach every q/k/v/o/gate/up/down projection, except a
  `v_proj` on a layer whose values are its keys (`use_k_eq_v`);
- `head_weights` reads a head, or refuses naming the architecture
  (an expected failure, reported as such).

A fixture copy of llama with a wrong residual fails the residual law by
name. A new architecture is admitted by adding its file, its tiny model
and its captured template, and passing the kit.
