# What a node fingerprint covers, and what it does not

A run that errors is a nuisance; a run that returns different numbers
with no signal is a corrupted finding that gets written up, cited and
built on. `resume.node_fingerprint` is what stands between us and the
second one, so this is an inventory of what it actually covers.

## The body that is hashed

```python
body = {
    "block":   block,          # "~canonical/ops/residuals/vectors/1"
    "params":  dict(params),   # serialize_params(): DECLARED params only
    "inputs":  input_hashes,   # content hashes of upstream node outputs
    "compute": core_version,   # mechbench_compute.__version__
    "model":   model,          # the wire form of the model ref
}
```

Serialized with `mechbench_schema.dump_canonical` (canonical CBOR), so
param ORDER does not matter — two identical param sets built in
different orders are one computation.

## Covered

- **The block identity**, including its version suffix.
- **Every declared param**, including ones added later: a protocol
  that starts passing `pool` gets a different fingerprint than one
  that does not.
- **Upstream content**, by hash. A changed corpus is a changed node.
- **The model**, in its canonical wire form (`{base, adapters}`), so an
  adapter swap is visible.
- **The compute version** — see the next two sections, which are the
  whole point of this document.

## An extension's operation: the pin is the block

A core operation's `block` is its stored path, and `compute` is its
lever. An extension's operation has no such lever — its code ships
outside compute — so its `block` is hashed in its pinned spelling,
`<owner>/<project>/ops/<family>/<leaf>@sha256:<digest>`, where the
digest is that of the extension version that provided it
(`Resolved.pinned`; the digest rule is in `docs/OPS_LAYOUT.md`). Run
creation writes that spelling into the flattened graph; a graph that
still names the bare address, or `@<n>`, is resolved against the
installed version first and fingerprinted by its pin, so the bare
spelling and its pin are one computation, and a different version of
the extension is a different one. A pin that does not match the
installed version does not run. Core fingerprints are unchanged by
this: their spelling is still `~canonical/ops/<name>`.

## NOT covered

- **Unstated defaults.** Params are hashed as declared. Change
  `DEFAULT_BRIDGE_SIGMA` from 2.0 to 2.5 and every protocol that omitted
  `bridge_sigma` keeps its fingerprint. This is deliberate — hashing
  resolved defaults would churn every fingerprint whenever a signature
  is touched — but it means the version string carries that weight.
- **Block semantics.** There is no per-block semantics version. Change
  what a block *does* without changing its name or its declared params
  and only `compute` notices.
- **The environment** beyond the compute version: MLX version, chip,
  numerics. Those are recorded as hardware/numerics lineage but are
  not part of process identity.

## The version on a source tree

`core_version` is `mechbench_compute.__version__`, which reads the
installed distribution's **metadata**. Under `pip install -e`, that
metadata only updates when somebody reinstalls — so the code can run
arbitrarily far ahead of the version it claims, and `importlib.metadata`
can resolve a different dist depending on the working directory.
Combined with "unstated defaults are not covered," a stale string would
be the only thing standing between a changed block default and a
silently reused partial.

So when the imported package is *not* inside `site-packages` /
`dist-packages` — i.e. it is a source tree someone can edit — the
version gains a digest of the `.py` content actually on disk:

```
0.37.0+src.dcd3217384cc
```

Content, not mtime (a git checkout touches files it did not change, and
a spurious fingerprint miss costs real compute). Paths are hashed too,
so moving code between files counts. It costs ~2 ms, once, at import.
A released wheel is not a source tree and keeps its plain version; the
release gate asserts that.

The consequence to expect: **on a development machine, editing any
compute source file invalidates every resume partial.** That is the
correct behaviour and it is cheap — it is also why the digest is
content-based rather than a timestamp.

## Resume levels, and why `satisfies` reads backwards

`satisfies(offered, required)` looks inverted until you notice a level
is a **demand**, not a guarantee:

- `restart` satisfies everything — it recomputes, so there is no
  partial to mistrust.
- `state-restorable` ranks *with* `reproducible`, because full state
  capture is bit-identical.
- `exchangeable` does not satisfy `reproducible`.

Both of those are deliberate. Tests in `tests/test_fingerprints.py`
pin them so the next reader does not have to re-derive the polarity.

An operation declares its level in its `OP` (`resume=Resume(level,
items=)`; one that writes none is `reproducible`, items off), and one whose level depends on its params also defines
`read_resume_level(params, inputs)` in its file, which `resume_level`
asks first. A block that is not an operation is `restart` — safe by
omission — and so is every operation that declares it: `activations/capture`
and `geometry/span` among them, so no partial of theirs is ever reused.

## The rule this leaves

> If a change alters results without raising, it must move a
> fingerprint.

The only lever for "semantics changed" is the compute version, so that
rule reduces to: **bump the version in the same commit as the behaviour
change.** On a dev machine the source digest enforces it automatically;
on a released install it is a human promise, which the release notes
keep.
