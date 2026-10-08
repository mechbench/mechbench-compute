# The plugin API

`mechbench_compute.api` is the surface an operation is written against,
whether it is core's or an extension's. An operation's file imports
mechbench from there and nowhere else; the standard library, numpy,
`mlx` (through `mechbench_compute._mlx`) and third-party packages are
its own business.

## What is in it

`api.__all__` is the list. In short:

- **`Context`**, what the executor lends an operation for one node:
  `model(ref)`, `loaded`, `sub(target, inputs, params, *, budget,
  on_item, on_start)`, `provider(model_ref)`, `memo(key)`,
  `materialize(label)`, `evict_model()`, `secrets`, the progress and
  resume callbacks. `NeedNotDeclared` is what a member raises when the
  operation did not declare the need that lends it; `ProviderClient` is
  what `provider` returns (`chat`, `embed`).
- **The model surface**: `Model`, `HookInfo`, `Capture`, `Patch`,
  `Ablate`, `compose`, `plan_intervention`, `compile_intervention`,
  `edit_weights`, `sample_completion_cached`, `render`, `encode`,
  `prefill_decision`, `TokenReadout`, `points`.
- **The reads**, for an operation that reads the model's decision as
  core's do: `resolve_target`, `read_answer` (a token by its text or by
  id), `resolve_outcomes`, `read_last_logp`, `read_metric` with
  `METRICS`, `OUTCOME_METRICS` and `METRIC_DOC`, `report_own_top1`,
  `read_record_coords`, `positions` (the position selector),
  `resolve_layers`, `check_written` (refuses a write the checkpoint does
  not make with a `PointRefused`, code `POINT_ABSENT`), `read_mask`
  (dimensions, a direction or a frame, as an operator's `mask` is read),
  `add_to_span` (what a node did, a backward pass among it, counted in its
  span), `read_f32` (any backend's array as float32 numpy),
  `read_array_ops` (the array arithmetic for the backend an array lives
  on, a gradient by additive deltas at named points among it) and
  `MAX_VECTOR_FLOATS`.
- **The lexicon helpers**: `collection`, `items_of`, `item_kind_of`,
  `read_header`, `arch_header`.
- **The tensor store**: `ShardWriter(dir)` takes items one at a time —
  each with a `vector`, any further per-item arrays (numpy, one width
  each across the collection), numbers and JSON fields — and writes them
  as safetensors shards; `tensor_collection(item_kind, writer.close(),
  **header)` is the collection that names them. The executor uploads
  the shards beside the object, and `items_of` reads them back a shard
  at a time. This is how a result past the 64 MiB object limit is
  stored.
- **The dictionary helpers**, for an operation that reads through a
  `direction/dictionary`: `read_dictionary_weights` (its encoder,
  decoder, biases and thresholds as arrays, refusing what is not a
  sparse autoencoder), `read_dictionary_activations` (a model's
  activations at the point and layer it reads, one forward pass) and
  `encode_features` (activations through its nonlinearity, one value
  per feature).
- **The record helpers**, for comparing records as `records/diff` does:
  `flatten_record` (a record as dotted paths), `index_by_key` (records
  by id or named coordinates, a duplicate refused), `is_field_match` (a
  path against a field pattern), `MOVING_FIELDS` (the fields that move
  on every run), `read_numbers` (a number or a nested list of numbers as
  one flat list, None otherwise), and `compute_version`.
- **The declaration**: `Op`, `P`, `In`, `Output`, `Otherwise`,
  `Resume`, `Kind`, `Draw`, `Metric`, `Notable`, `F`, and
  `read_body_level` / `read_model_level` for an operation whose resume
  level depends on its params. A kind's `notable`, `Notable(field, key,
  metric, threshold)`, says how a try's result of that kind reads
  against its baseline: the field compared, the item fields that match
  an item to its counterpart, the metric (`difference`, or one the kind
  declares) and the threshold the metric must pass. A try's `notable`
  carries the raw changes (before, after, difference, the metric's
  distance, the floors and the floor, the threshold) and one categorical
  `state`: `noise` within `k` floors, `small` past them but under the
  threshold, `moved` past both; without a floor the threshold alone
  judges. A kind without one has no notable.
- **The extension**: `Extension` and `Package`, the `MANIFEST` an
  extension package's entry point (`mechbench.extensions`) names; its
  `ops/` and `kinds/` are walked as core's are (docs/OPS_LAYOUT.md).

Importing `mechbench_compute.api` loads nothing; each name loads its
module the first time it is read, so a declaration imports cleanly on a
machine without MLX.

## Short names inside an extension

An extension writes its own kinds and core's by short name,
`family/leaf`, in its declarations and in its code alike.

- **At registration** the registry rewrites every kind an extension's
  `Op` or `Kind` names short — a port, an output, an `Otherwise`, an
  `extends` — to a full address: one of the extension's own kinds
  becomes `<owner>/<project>/kinds/<family>/<leaf>`, a core kind stays
  its core name. `check_graph` and `check_inputs` only ever see full
  addresses. The registered op's and kind's `to_dict()["path"]` is its
  address; the package's own manifest (`Extension.to_dict()`) keeps the
  author's short `name` and carries the address as `path`.
- **While an extension's operation runs**, its `Context.scope` is the
  extension's `<owner>/<project>`, and the registry knows that scope
  for the duration of `run`. `collection(kind, …)`, `resolve_kind`,
  `satisfies` and `qualify_kind` read a short kind name in that scope:
  core first, then the extension's own kinds. Each also takes an
  explicit `scope=` for code that runs outside `run`. A core
  operation's scope is `None`, and a sub-run's core nodes run in
  none. `items_of` reads a collection, not a kind name, so there is
  nothing for it to resolve.
- **Core wins, so a collision is refused.** An extension that declares
  a kind whose short name is a core kind's is refused at load, as an
  owner that is a core family name is: a short name must mean one
  thing.

## The promise

- The api is versioned with compute's minor release.
- A change that breaks a name in it — removed, renamed, a parameter
  that means something else — is a minor bump and a line under
  "Changes that raise" in CHANGELOG.md. Adding a name is not a break.
- An extension's `min_compute` names the compute release whose api it
  was written against.
- Anything not in `api.__all__` may change in any release without a
  word.

## Core is a client of it

`tests/test_ops_use_the_api.py` holds every file under `ops/` to the
rule. The operations that still import internals directly are listed
there in `IMPORTS_INTERNALS`; the list may only shrink, and a file that
no longer needs to be on it must come off.

## Conformance

`mechbench_compute.conformance` is the declaration suite as functions:
the same rules that hold core's operations and kinds in the test suite,
run over an extension's manifest. `extension test` runs it in the
author's shell; the verification job runs it on a runner, in a scratch
environment beside a named compute release.

```
python -m mechbench_compute.conformance <package | module:ATTR | entry point> [--inputs DIR] [--model]
```

prints the report as JSON and exits 1 when a declaration fails or an
example is refused. `--inputs DIR` reads each example's `$ref` inputs
from `DIR/<bench path>.json` (or `DIR/<file>`); `--model` says a model
is available, so operations that need one run too. The same pieces,
imported:

- `check_manifest(extension)` takes an `Extension` or its wire form
  (`Extension.to_dict()`, or the JSON the platform stores) and returns
  a list of `Finding(code, at, message, severity)`. It never raises.
- `check_package(extension)` reads the loaded package: each operation
  module's `run`, its function names, the `ctx` members its `run`
  reaches for, the kind literals in its source, and whether its closed
  sets of words are enforced in code.
- `run_examples(extension, *, resolve_inputs, model=False)` installs
  the extension into the registry for the duration, resolves each
  operation's `example_inputs` through `resolve_inputs(op, example_inputs)`
  (a callback returning collections keyed by port), runs the example
  twice in-process, checks the output against the declared kind and
  its key fields, and compares the two results' canonical bytes.
- `check_extension(...)` does all three and returns a `Conformance`,
  whose `to_dict()` is the extension object's `conformance`:
  `{compute, declarations: passed | failed, double_run: identical |
  differs | skipped, installs_beside}`. `verified_by` and `at` are the
  platform's to write. `report()` adds the findings and each example's
  result.

A finding's `at` is the operation or kind it is about, then `.param`,
`:port` or `#function` where it is narrower. `severity` is `error` or
`warning`; any error fails the declarations.

| code | what it means |
|---|---|
| `NO_SUMMARY`, `NO_DESCRIPTION` | the operation says nothing about itself |
| `NO_OUTPUT`, `NO_OUTPUT_DOC` | it emits nothing, or does not say what it emits |
| `PARAM_UNTYPED`, `PARAM_UNDOCUMENTED`, `PARAM_DUPLICATE` | a param lacks a type or a description, or is declared twice |
| `PORT_UNDOCUMENTED`, `PORT_DUPLICATE`, `PORT_NAME_INVALID` | a port lacks a description, is declared twice, or is not `[a-z][a-z0-9_]*` |
| `PORT_KIND_UNKNOWN` | a port names a kind neither core nor the extension declares |
| `PARAM_NAMES_PORT` | a name is both a param and a port, or prose describes an input as a param |
| `READS_UNKNOWN`, `REPLACES_UNKNOWN`, `READS_NOT_EXPRESSION`, `READS_UNSAID` | an expression's `reads`/`replaces` are wrong, or an expression on several ports does not say which it reads |
| `SUMMARY_RESTATES_REF`, `SUMMARY_NOT_SENTENCE`, `SUMMARY_TOO_LONG` | the summary is not one sentence for a stranger under 400 characters |
| `HOUSE_IDIOM` | published text names something only its author can follow (a ticket number, an internal repo) |
| `EXAMPLE_REFUSED` | the example names a param or port the operation does not have, or leaves a required port unwired |
| `EXAMPLE_TYPE_MISMATCH` | an example value does not fit its param's declared type |
| `PARAM_REDECLARES_COMMON` | a param redeclares one every operation takes |
| `NAME_INVALID`, `NAME_DUPLICATE` | the name is not `family/leaf`, or two operations share it |
| `OP_NOT_VERB`, `OP_NAMES_KIND` | the leaf does not start with a verb, or the operation shares a kind's name |
| `NEED_UNKNOWN` | a need outside the vocabulary |
| `NOT_JSON` | the declaration has no JSON form |
| `TYPE_INVALID`, `OBJECT_UNDECLARED`, `FIELDS_INVALID` | a type outside the grammar; an `object` with no fields; fields on a type with no `object` |
| `VALUE_INVALID`, `CHOICES_INVALID` | a `value=` that is not a declared grammar; choices that do not fit the type or the default |
| `OTHERWISE_INVALID` | an `Otherwise` names an unknown kind, param or port, or the output prose does not name it |
| `OUTPUT_KIND_UNKNOWN`, `OUTPUT_NOT_COLLECTABLE` | the output kind is undeclared, or a collection of a kind with no key |
| `KIND_NAME_INVALID`, `KIND_DUPLICATE`, `KIND_NOT_DESCRIBED`, `KIND_FIELD_UNDESCRIBED` | a kind's name, uniqueness, summary or fields |
| `KIND_REQUIRED_UNKNOWN`, `KIND_EXTENDS_UNKNOWN`, `KIND_CYCLE` | a required field nobody declares; an unknown parent; a cycle |
| `NO_SPEAK` | an extension kind with no `speak` |
| `KIND_NO_EXTENDS`, `KIND_SEALED_FAMILY`, `KIND_SHADOWS_CORE`, `KIND_CHANGES_ANCESTOR` | an extension kind extends nothing, sits in a sealed family, takes a core kind's name, or changes a field or header field a core ancestor declares |
| `NO_RUN` | an operation module defines no `run` |
| `NAME_NOT_VERB` | a function in an operation module does not start with a verb |
| `NEED_UNDECLARED` | `run` reaches for a `ctx` member no declared need lends |
| `NEED_UNUSED` (warning) | a declared need `run` never reaches for |
| `KIND_LITERAL_UNKNOWN` | source names an undeclared kind in a `"kind": "…"` literal |
| `CLOSED_SET_UNENFORCED` | declared choices no code enforces as a set |
| `NO_EXAMPLE`, `EXAMPLE_NEEDS_MODEL` (warnings) | no example to run; an example that needs a model where none is available |
| `EXAMPLE_INPUTS_UNRESOLVED`, `EXAMPLE_FAILED` | the inputs could not be read; the run raised |
| `EXAMPLE_WRONG_KIND` | the output does not satisfy the declared kind, or an item lacks the kind's key fields |
| `EXAMPLE_DIFFERS`, `EXAMPLE_NOT_CANONICAL` | the two runs differ (an error when the operation declares itself deterministic); the output has no canonical bytes |
| `EXTENSION_REFUSED` | the registry refused the extension at load |

Core's own test files (`test_lexicon.py`, `test_lexicon_shapes.py`,
`test_kinds.py`) assert that `check_core()` has no findings, so a rule
lives in one place. The exceptions core keeps are allowed codes per
operation, in the test files: the tools emit nothing, and a short list
of operation names that predate the verb rule. The kinds that do not
speak yet are core's ratchet in `test_kinds.py`, not a finding.
