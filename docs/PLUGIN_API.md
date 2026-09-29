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
- **The lexicon helpers**: `collection`, `items_of`, `item_kind_of`,
  `read_header`, `arch_header`.
- **The declaration**: `Op`, `P`, `In`, `Output`, `Otherwise`,
  `Resume`, `Kind`, `Draw`, `Metric`, `F`, and `read_body_level` /
  `read_model_level` for an operation whose resume level depends on its
  params.
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
