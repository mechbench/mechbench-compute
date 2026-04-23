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
