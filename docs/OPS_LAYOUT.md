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
  parameters are written down, and it is complete here: a param several
  operations declare the same way, a port they share, a paragraph of
  prose — each is written out in the file rather than imported, so the
  contract reads whole without opening anything else. The only thing
  the file takes from `mechbench_compute.lexicon` is the vocabulary a
  declaration is written in — `Op`, `P`, `In`, `Output`, the kinds.
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

While the move is under way, the module a helper left is that package's
`__init__.py` and imports the helper back, so nothing that named it
through the module breaks. An operation's file is different: it is a
leaf, nothing imports back from it, and whatever named its contents
through their old module was rewritten to name the new one.

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
- **Whether it can be computed in chunks, and how** — its file defines
  `MONOID`. That one line is what the reduce machinery, the chunk
  harness and the resume levels all used to read from three separate
  tables, kept in step by hand.

A test that wants to stand in for an operation patches the operation's
own file — `monkeypatch.setattr(fill, "run", flaky)`,
`monkeypatch.setattr(total, "MONOID", Bad)` — because that is where it
runs from. Patching a table it used to be listed in changes nothing.

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
pipeline.py       walking the graph: order, resume, emission
dispatch.py       how a node reaches an operation's run()
model.py          loading weights, fusing adapters
remote.py         the nodes a provider answers, run in a wave
tools.py          what a model may call mid-turn
chat.py           the local half of text/chat
memo.py           a node's memo of the remote calls it made
legacy_kinds.py   the two spec kinds that came before the graph
```

What they share is one definition per file, as `serialize_params` and
`read_tokenizer_id` already were. The class is still one class, and its
methods and their call sites are unchanged: a mixin is how a method keeps
its `self`.
