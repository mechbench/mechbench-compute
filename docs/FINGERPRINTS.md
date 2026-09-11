# What a node fingerprint covers, and what it does not

Task 000433. A run that errors is a nuisance; a run that returns
different numbers with no signal is a corrupted finding that gets
written up, cited and built on. `resume.node_fingerprint` is what
stands between us and the second one, so this is an inventory of what
it actually covers — written after auditing it, not from the design
intent.

## The body that is hashed

```python
body = {
    "block":   block,          # "~canonical/ops/residuals/vectors/1"
    "params":  dict(params),   # _wire_params(): DECLARED params only
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
  numerics. Those are recorded as hardware/numerics lineage (000402)
  but are not part of process identity.

## Why `compute` had to be fixed (2026-09-11)

`core_version` is `mechbench_compute.__version__`, which reads the
installed distribution's **metadata**. Under `pip install -e`, that
metadata only updates when somebody reinstalls — so the code can run
arbitrarily far ahead of the version it claims.

This was not hypothetical. Measured on this project the same day:

- the runner's repo venv reported **0.20.0 while executing 0.36.0
  code** — sixteen versions of drift;
- the same interpreter reported **0.37.0 from the source repo and
  0.36.0 from anywhere else**, because `importlib.metadata` resolved a
  different dist depending on the working directory.

Combined with "unstated defaults are not covered," that made a stale
string the only thing standing between a changed block default and a
silently reused partial.

**The fix:** when the imported package is *not* inside `site-packages`
/ `dist-packages` — i.e. it is a source tree someone can edit — the
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

Both of those are deliberate and documented in `resume.py`. Tests in
`tests/test_fingerprints.py` pin them so the next reader does not have
to re-derive the polarity.

A block absent from `BLOCK_RESUME` is `restart` — safe by omission.
`residuals/vectors` and `vectors/mst` are both absent, so no partial of
theirs is ever reused.

## The rule this leaves

> If a change alters results without raising, it must move a
> fingerprint.

Today the only lever for "semantics changed" is the compute version, so
that rule reduces to: **bump the version in the same commit as the
behaviour change.** On a dev machine the source digest now enforces it
automatically; on a released install it is still a human promise, which
is what task 000434's two-list release notes are for.
