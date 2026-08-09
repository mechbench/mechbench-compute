# mechbench-core

The compute engine for [mechbench](https://github.com/mechbench/mechbench) — composable mechanistic-interpretability primitives built on MLX.

This repo provides:

- **Hook-aware forward pass.** One canonical path through the model; instrumentation via named hook points and TransformerLens-style callbacks.
- **Declarative interventions.** `Ablate`, `Capture`, `Patch` primitives composable into a single `model.run(..., interventions=[...])` call.
- **Activation cache.** `ActivationCache` container for collected activations; bf16 throughout, float32 only at the analysis boundary.
- **Architecture adapter.** `Arch` dataclass that handles per-variant differences (layer count, global-attention pattern, RoPE parameters, etc.). Currently supports Gemma 4 E4B and E2B; the adapter pattern follows TransformerLens 3's `TransformerBridge`.
- **Analysis helpers.** Logit lens, direct logit attribution (`accumulated_resid`, `decompose_resid`, `head_results`, `logit_attrs`), fact vectors, centroid decoding, probe primitives, head-weight static analysis, geometry metrics.
- **Plot helpers.** Matplotlib conventions baked in for quick diagnostic figures — not the full visualization surface (that lives in `mechbench-ui`).

See [`PACKAGE_README.md`](PACKAGE_README.md) for the full API tour and worked examples.

## Install

```bash
pip install mechbench-core
```

From source:

```bash
git clone https://github.com/mechbench/mechbench-core.git
cd mechbench-core
pip install -e '.[dev]'
```

Apple Silicon required (MLX is the only supported backend today). A PyTorch backend would live as `mechbench_core.backends.torch` alongside the MLX one if/when the need arises; splitting repos by backend is explicitly not planned.

## Quick start

```python
from mechbench_core import Model, Ablate, Capture

model = Model.load()
ids = model.tokenize("Complete this sentence with one word: The Eiffel Tower is in")

result = model.run(ids)
for tok, p in result.top_k(model.tokenizer, k=5):
    print(f"{tok!r:20s} p={p:.4f}")
```

## Distributional-target training (`distill` + `lora`)

Primitives for training a model toward a specified *distribution* over
responses rather than toward example responses (task
[`000114`](https://github.com/mechbench/mechbench/blob/main/tasks/mechbench-core/open/000114-entropy-reward-finetuning.md)):
soft-target cross-entropy at decision tokens has gradient P − T, so the
adapter learns to *emit the distribution*.

```python
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mechbench_core import Model, distill, lora
from mechbench_core.distill import TargetMap

model = Model.load()
tok = model.tokenizer

# A target is a Map<String, Double> — hardcoded, from JSON, or uniform —
# with whole-map transforms that each return a new map:
target = TargetMap.from_json("weights.json").sqrt().normalize()
target = TargetMap.uniform([str(i) for i in range(1, 7)])   # fair d6

# Compile it against the rendered prompt: items become token paths
# (multi-token items share trie nodes; a closer appends continuation
# anchors so the flattening can't leak past the envelope):
prompt = distill.render_chat(tok, system, "Please roll the die.",
                             prefill='{ "roll": ')
trie = target.tokenize(tok, prompt, closer=" }")

n = lora.apply_lora(model.lm)                 # freeze + wrap q/v projections
step = nn.value_and_grad(model.lm, distill.soft_ce)
opt = optim.Adam(learning_rate=1e-4)
rng = np.random.default_rng(7)
for _ in range(steps):
    batch = [trie.hard_example(trie.sample(rng)) for _ in range(3)]
    batch.append(trie.marginal_example())     # exact first-token marginal
    batch.append(sharp_anchor)                # keeps confident tasks sharp
    loss, grads = step(model.lm, batch)
    opt.update(model.lm, grads)

lora.save_adapter(model.lm, "adapter.safetensors")
# Later, on a fresh model: merge + exact undo
handle = lora.fuse(model.lm, lora.load_adapter("adapter.safetensors"),
                   scale=16 / 8)              # alpha / rank from training
lora.restore(model.lm, handle)
```

Calibration is measured at item level (`trie.score`, `distill.item_metrics`
— captured mass, entropy, KL from target) and at the decision token
(`distill.first_token_metrics`). `python -m mechbench_core._smoke_distill`
runs the full lifecycle on E2B.

**Two forward paths.** Training and scoring call `Model.lm` (the text
decoder, uniform across families) directly — plain module calls,
differentiable, no instrumentation. `Model.run` remains the hook-aware
forward for capture/patch/lens work. Adapters bridge the two: `fuse` an
adapter into the weights and every instrumented run sees the adapted
model; `restore` flips it back, so base-vs-adapted comparisons run in one
script.

## Status

Lifted from `gemma4-mlx-interp/gemma4_mlx_interp/` (the predecessor repo, since renamed to `mechbench-experiments`). The `Arch` adapter now supports both Gemma 4 E4B and E2B; generalization to other architecture families is ongoing — track open work in the meta repo's [`tasks/mechbench-core/`](https://github.com/mechbench/mechbench/tree/main/tasks/mechbench-core) directory.

The substrate epic that will define how intermediate results are cached and shared across experiments is [`000162`](https://github.com/mechbench/mechbench/blob/main/tasks/mechbench-core/open/000162-dag-solver-and-content-addressed-memoization-cache.md) (DAG solver + content-addressed memoization). It consumes the canonical-serialization guarantee from [`000161`](https://github.com/mechbench/mechbench/blob/main/tasks/mechbench-schema/open/) (binary formats) and the path grammar from [`000163`](https://github.com/mechbench/mechbench/blob/main/tasks/mechbench-meta/open/) (identity scheme).

## Relationship to other mechbench repos

- **`mechbench-schema`** — the typed emission contract. `mechbench-core` emits records shaped by schema types; currently a soft dependency as the emission layer is formalized.
- **`mechbench-experiments`** — research scripts and findings that consume this package. Uses `mechbench-core` as its primary dependency.
- **`mechbench-agent`** — exposes these primitives as agent-callable tools. Imports `mechbench-core`.
- **`mechbench-remote`** — wraps `mechbench-core` behind an RPC contract for remote (H100-class) compute. Imports `mechbench-core`.
- **`mechbench-ui`** — TypeScript frontend. Does not import `mechbench-core` directly; reads bundles produced by it through the `mechbench-schema` contract.

See the [meta repo](https://github.com/mechbench/mechbench) for the family overview and [PHILOSOPHY\_AND\_DIRECTION.md](https://github.com/mechbench/mechbench/blob/main/docs/PHILOSOPHY_AND_DIRECTION.md) for the design principles.

## License

MIT.
