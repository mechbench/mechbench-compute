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
  `check_params` enforces. It is the only place the operation's
  parameters are written down.
- **`run(ctx, inputs, params)`** is the one entry point, the same
  signature for every operation. It unpacks `inputs`, asks `ctx` for
  what it needs, and calls the mechanism.
- **The mechanism** is one or more plain functions that take loaded
  things — a model, a list of records — and return the result. Tests
  call these directly.
- **`MONOID`**, when the operation can be computed in chunks: the class
  that reduces them. Its presence is the declaration.

An operation's file holds what only that operation uses. What several
operations share stays in a module named for what it is:
`mechbench_compute/intervene.py` is the intervention grammar, which
`intervene/apply`, `intervene/steer` and `text/generate` all compile
specs with; `mechbench_compute/chat.py` is what `text/chat` and
`eval/judge` both talk to a model through. An operation's file imports
from those by name, so the import line says what the shared thing is.

There is deliberately no `_common.py`. Filing shared code by *who uses
it* rather than *what it is* was tried first and measured: once the
dependencies between modules were followed honestly, one grab-bag came
to 1,614 lines and 62 definitions — the file this layout exists to
prevent, under a name that says nothing.

## What `ctx` offers

`ctx` is `mechbench_compute.ops.Context`: what the executor lends an
operation for the duration of one node. Twenty-seven of the thirty-three
operations that need the executor need exactly one thing from it.

| | |
|---|---|
| `ctx.model(ref)` | the loaded model for a `model` param |
| `ctx.on_item`, `ctx.on_start` | progress: one call per item, one when the count is known |
| `ctx.resume_items`, `ctx.resume_state` | what an interrupted run already finished |
| `ctx.on_checkpoint` | for an operation that checkpoints (`adapter/train`) |
| `ctx.secrets` | the owner's provider credentials |
| `ctx.input_paths` | the stored label each input arrived from |
| `ctx.bindings`, `ctx.result_base` | the run's bound params, and where its results land |
| `ctx.executor` | the executor itself — for the few operations that are control flow (`records/map`, `records/fold`) or that serve two tiers (`text/chat`) |

Every field has a default, so a test builds one in a line:
`run(Context(model=fake), inputs, params)`.

## What is derived, and so is not written down

- **The registry.** `mechbench_compute.ops` imports every module under
  it. There is no table of operations to keep in step with the files.
- **Whether an operation is pure** — `OP.requires == "pure"`.
- **Whether it can run with no executor at all** — it is pure and its
  `run` never reads `ctx`. That is the set a tool handler or a chunked
  reduce may call.
- **Whether the executor fuses an adapter around it** — it requires
  local weights (`mlx-local`) *and* declares an `adapter` input port:
  there are weights to fuse onto, and an adapter may arrive. The port
  alone does not say so — `adapter/measure` and `adapter/publish` take
  an adapter as the thing they operate on.

## Two rules the layout depends on

**An operation's file imports cleanly on a machine that cannot run it.**
The lexicon is read by the documentation build, by the API, and by
runners without Apple silicon. `mlx` comes from
`mechbench_compute._mlx`, which raises a clear error on first use rather
than at import.

**No file is over the size budget.** A gate in the suite holds it. A
3,000-line file is not something an agent can read to understand one
operation, and nothing else keeps a file small.
