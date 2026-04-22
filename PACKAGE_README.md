# mechbench_core

A small mechanistic-interpretability framework for Google's Gemma 4 E4B running locally on Apple Silicon via MLX. Layered on top of `mlx-vlm`; owns one canonical forward pass and exposes it through a clean API for ablation studies, activation patching, logit lenses, and fact-vector geometry.

This is project-specific: the architectural constants (42 layers, global layers at 5/11/17/23/29/35/41, 2560-dim residual, MatFormer side-channel) are hard-coded. If you're here to probe a different model, you'll need to generalize `_arch.py` first.

## Quick start

```python
from mechbench_core import Model

model = Model.load()  # loads mlx-community/gemma-4-E4B-it-bf16 from HF cache
ids = model.tokenize("Complete this sentence with one word: The Eiffel Tower is in")
result = model.run(ids)

for tok, p in result.top_k(model.tokenizer, k=5):
    print(f"{tok!r:14s} p={p:.4f}")
# 'Paris'         p=0.9770
# ' Paris'        p=0.0150
# ...
```

`Model.run` always returns a `RunResult` with `.logits` (`[1, seq_len, vocab_size]` bf16) and `.cache` (an `ActivationCache`, empty unless you requested captures).

## What's in the box

There's a rough dependency gradient — forward-pass + hook machinery is the foundation, and interventions, analysis helpers, and plot helpers sit on top. Every piece can be used directly; the grouping is just a navigation aid.

| Module surface | What you get | When you'd use it |
|:-----:|-------------|-------------------|
| **Forward + hooks** | `Model.run(input_ids, hooks={}, capture=[])` — TransformerLens-style callbacks at 294 named hook points | You're writing a custom intervention that the declarative ones don't support |
| **Interventions** | `Ablate` / `Capture` / `Patch` composable in a list | Day-to-day experiment scripts. Most code lives here |
| **Prompts, lens, geometry** | `Prompt` / `PromptSet` / `validate`, `logit_lens_*`, `fact_vectors*`, `centroid_decode`, geometry stats | Analyses that need the standard scaffolding — logit lens trajectories, fact-vector clustering, etc. |
| **Plot helpers** | `bar_by_layer`, `lens_trajectory`, `position_heatmap`, `pca_scatter`, `similarity_heatmap` | You want the project's plot conventions (red=global, blue=local, dashed lines at global layers) without rewriting them |

---

## Forward pass + hooks

```python
from mechbench_core import Model, all_hook_names

model = Model.load()
print(len(all_hook_names()))  # 294 — 42 layers × 7 points each
```

### Hook points (layer-scoped)

Every layer `i` in `[0, 42)` exposes:

```
blocks.{i}.resid_pre           # layer input (== resid_post[i-1] for i>0)
blocks.{i}.attn_out            # attn branch output after o_proj + post_attention_layernorm
blocks.{i}.mlp_out             # MLP branch output after post_feedforward_layernorm
blocks.{i}.gate_out            # MatFormer per-layer-input gate output
blocks.{i}.resid_post          # layer output after layer_scalar
blocks.{i}.attn.weights        # post-softmax attention, [B, n_heads, L, S_kv]
blocks.{i}.attn.per_head_out   # weights @ values, [B, n_heads, L, head_dim]
```

The two `attn.*` points require the manual-softmax path instead of MLX's fused SDPA kernel. The framework auto-switches per-layer when you touch one of them. Other layers stay on the fast path.

### Raw hook API

A hook is a callable `(activation, info) -> mx.array | None`. Return `None` to pass through, or an mx.array of the same shape/dtype to replace:

```python
import mlx.core as mx

def zero_layer_14_mlp(act, info):
    return mx.zeros_like(act)

result = model.run(ids, hooks={"blocks.14.mlp_out": zero_layer_14_mlp})
```

Captures save the post-hook value into `result.cache`:

```python
result = model.run(ids, capture=["blocks.23.attn.weights", "blocks.14.mlp_out"])
weights = result.cache["blocks.23.attn.weights"]   # [1, 8, seq_len, seq_len] bf16
mlp_out = result.cache["blocks.14.mlp_out"]        # [1, seq_len, 2560]       bf16
```

Typos raise `InvalidHookName` with a `difflib.get_close_matches` suggestion. Out-of-range layer indices raise `LayerIndexOutOfRange`. Looking up a missing cache key raises `CacheKeyError` with the same style of suggestion.

---

## Declarative interventions

Pass a list of `Intervention` objects as `interventions=[...]`. Each compiles to one or more hook callbacks plus optional captures, and the framework composes them:

```python
from mechbench_core import Ablate, Capture, Patch
```

### Ablate — zero out components

```python
Ablate.layer(14)                   # skip the whole layer (resid_post = resid_pre)
Ablate.attention(23)               # zero attn_out; keep MLP + gate
Ablate.mlp(14)                     # zero mlp_out; keep attn + gate
Ablate.head(29, head=7)            # zero one head's slice of attn.per_head_out
Ablate.side_channel()              # zero MatFormer gate at every layer
Ablate.side_channel(layers=[11, 17, 23])  # or at a specific subset
```

### Capture — save activations

```python
Capture.residual(layers=range(42))         # all resid_post
Capture.residual(layers=[10, 20, 30], point="pre")  # resid_pre at picks
Capture.attn_weights(layers=[23, 29])      # forces manual-attn at these layers
Capture.gate_out(layers=range(42))
Capture.per_head_out(layers=[29])
```

### Patch — replace activations

```python
# 1. Capture a clean run
clean = model.run(clean_ids,
                  interventions=[Capture.residual(layers=range(42))])

# 2. Run corrupt and swap in clean resid_post at one (layer, pos)
result = model.run(corrupt_ids, interventions=[
    Patch.position(layer=10, position=13, source=clean.cache),
])

# Or patch with an arbitrary tensor
Patch.activation(layer=10, position=13, value=my_tensor)
```

### Composition

Multiple interventions targeting the same hook point chain in list order. Each callback receives the previous return value and may pass through or modify:

```python
result = model.run(ids, interventions=[
    Ablate.head(29, head=7),            # zero head 7
    Capture.per_head_out(layers=[29]),  # capture the post-ablation tensor
])
captured = result.cache["blocks.29.attn.per_head_out"]
# captured[:, 7, :, :] is all zeros; other heads have signal.
```

---

## Prompts, lens, geometry

### Prompt + PromptSet

```python
from mechbench_core import Prompt, PromptSet

my_set = PromptSet(name="my_battery", prompts=(
    Prompt(text="Complete this sentence with one word: The capital of France is",
           target="Paris", subject="France", category="capital"),
    Prompt(text="Complete this sentence with one word: The opposite of hot is",
           target="cold", subject="hot", category="opposite"),
))
```

Fields are all optional except `text`. `subject` is a substring used by `fact_vectors` to pick which token's residual to extract. `category` is for grouped analyses. `metadata` is a free-form dict.

This project's specific prompt collections live in `experiments/prompts/` (they're project data, not framework infrastructure):

```python
from experiments.prompts import (
    FACTUAL_15,              # 15 prompts used by step_01-09
    BIG_SWEEP_96,            # 12 categories × 8 prompts (step_12)
    STRESS_TEMPLATE_VAR,     # 4 phrasings × 4 countries (step_13)
    STRESS_CROSS_LINGUAL,    # 5 languages × 4 countries (step_13)
    STRESS_CREATIVE,         # 8 subjective / metaphorical prompts (step_13)
)
```

If you're using `mechbench_core` outside this project, you'd build your own `PromptSet` instances similarly.

### Validation

```python
validated = FACTUAL_15.validate(model)
#   [OK] Complete this sentence with one word: The Eiffel Tower is in ...
#   [SKIP] Complete this sentence with one word: The Amazon River flows ...
#   14 / 15 prompts validated.

for vp in validated:
    print(vp.prompt.text, vp.target_token, vp.baseline_lp, vp.confidence)
```

Keeps only prompts where confidence ≥ `min_confidence` (default 0.5) and, if `prompt.target` is set, the target matches the top-1 decoded token. For geometric experiments where you want to keep every prompt regardless of what the model says, pass `require_target_match=False`.

### Logit lens

```python
from mechbench_core import Capture, logit_lens_final, logit_lens_per_position, N_LAYERS

# Capture residual stream at every layer
result = model.run(ids, interventions=[Capture.residual(layers=range(N_LAYERS))])

# Final-position version: per-layer (rank, logprob) of a target token
ranks, logprobs = logit_lens_final(model, result.cache, target_id=paris_token_id)
# ranks.shape = (42,), logprobs.shape = (42,)

# Per-position version for step_08-style analyses
ranks_grid, logprobs_grid = logit_lens_per_position(model, result.cache, target_id)
# ranks_grid.shape = (42, seq_len), logprobs_grid.shape = (42, seq_len)
```

### Fact vectors + centroid decoding

```python
from mechbench_core import fact_vectors, fact_vectors_at, centroid_decode

# Single layer -> [n_prompts, 2560] float32
vecs = fact_vectors(model, validated, layer=30, position="subject")

# Multiple layers in ONE model pass per prompt -> {layer: ndarray}
vecs_by_layer = fact_vectors_at(model, validated, layers=[15, 30], position="subject")

# Decode the centroid of a group through the tied unembed (finding 11/12's technique)
capital_only = [v for v, cat in zip(vecs, labels) if cat == "capital"]
overall_mean = vecs.mean(axis=0)
top_tokens = centroid_decode(model, capital_only, k=8, mean_subtract=True,
                             overall_mean=overall_mean)
# [(' தலைநக', 0.104), (' city', 0.043), (' राजधानी', 0.049), ...]
```

### Pure-numpy geometry stats

```python
from mechbench_core import (
    cosine_matrix, intra_inter_separation,
    cluster_purity, silhouette_cosine, nearest_neighbor_purity,
)
```

Each takes numpy arrays; no model dependency.

---

## Plot helpers

Each helper takes numpy arrays plus an optional `ax` (creates a new figure if None), returns the `Axes`, and never calls `plt.show()` or `fig.savefig()` — the caller owns those.

```python
import matplotlib.pyplot as plt
from mechbench_core import (
    bar_by_layer,                     # red=global, blue=local bar chart
    lens_trajectory, logprob_trajectory,  # per-layer curves w/ geomean
    position_heatmap,                 # [layer × position] with markers
    pca_scatter, similarity_heatmap,  # for fact-vector geometry
)

fig, ax = plt.subplots(figsize=(14, 5))
bar_by_layer(mean_deltas, ax=ax,
             ylabel="mean Δ log p(target)",
             title="Layer ablation impact")
fig.savefig("caches/layer_ablation.png", dpi=140)
```

---

## Worked examples

### Layer ablation sweep (reproduces finding 02)

```python
import numpy as np
from mechbench_core import Ablate, FACTUAL_15, Model, N_LAYERS, bar_by_layer

model = Model.load()
valid = FACTUAL_15.validate(model)

loss_delta = np.zeros((N_LAYERS, len(valid)))
for i in range(N_LAYERS):
    ablation = Ablate.layer(i)
    for j, vp in enumerate(valid):
        r = model.run(vp.input_ids, interventions=[ablation])
        last = r.last_logits.astype("float32")
        # ... compute log p of target ...
        loss_delta[i, j] = ablated_lp - vp.baseline_lp

mean_delta = loss_delta.mean(axis=1)
bar_by_layer(mean_delta, ylabel="mean Δ log p", title="Layer ablation")
```

### Causal tracing (reproduces finding 09)

```python
from mechbench_core import Capture, N_LAYERS, Patch

clean = model.run(clean_ids, interventions=[Capture.residual(range(N_LAYERS))])

patch_probs = np.zeros((N_LAYERS, seq_len))
for L in range(N_LAYERS):
    for P in range(seq_len):
        r = model.run(corrupt_ids, interventions=[
            Patch.position(layer=L, position=P, source=clean.cache),
        ])
        patch_probs[L, P] = float(softmax(r.last_logits)[clean_answer_id])
```

### Fact-vector geometry (reproduces finding 10)

```python
from mechbench_core import (
    BIG_SWEEP_96, Model, PromptSet, centroid_decode,
    fact_vectors_at, nearest_neighbor_purity, similarity_heatmap,
)

valid = BIG_SWEEP_96.validate(model, min_confidence=0.0)
vecs = fact_vectors_at(model, valid, layers=[30])[30]
labels = np.array([vp.prompt.category for vp in valid])

nn_rate, _ = nearest_neighbor_purity(vecs, labels)  # 1.000 per finding 12
similarity_heatmap(vecs, labels)
```

---

## Notes on types and dtypes

- **Model weights and activations are bf16.** Cache is returned in bf16.
- **Always cast to float32 before going to numpy.** MLX → numpy on bf16 crashes with a PEP 3118 buffer format error. Use `x.astype(mx.float32)` or `cache.to_float32()` at the analysis boundary.
- **Model.run evals the cache for you.** You don't need to call `mx.eval()` on returned tensors; they're materialized before the function returns.
- **`layer.layer_scalar` is always `mx.ones((1,))` on E4B** in practice (we've checked), so multiplying by it is identity. Don't be surprised if `Ablate.layer` bypasses it via the resid_pre capture / resid_post patch trick rather than touching it directly.

---

## Smoke tests

Four self-contained smoke tests live next to the package:

```bash
python -m mechbench_core._smoke                # forward path: semantic top-1 on factual prompts
python -m mechbench_core._smoke_interventions  # composition: Ablate.head + Capture.per_head_out
python -m mechbench_core._smoke_plots          # plots: renders each helper on synthetic data
```

Plus an integration test in `experiments/smoke_analysis.py` (lives outside the framework because it depends on this project's specific prompt collections):

```bash
python experiments/smoke_analysis.py              # reproduces findings 01 / 11 / 12 numbers
```

Run any of them after a framework change to catch regressions.

---

## Scope / limitations

- **Gemma 4 E4B only.** Architectural constants are hard-coded in `_arch.py`. Other Gemma variants will need `N_LAYERS` / `GLOBAL_LAYERS` / `D_MODEL` / hook-point specifics updated, and the `_attention_with_internals` branch inspected for per-architecture details (GQA ratios, KV sharing, etc.).
- **No MoE, vision, or audio paths.** The 26B-A4B MoE variant uses a Router + Experts branch that `_forward.py` doesn't walk. Vision and audio towers are untouched.
- **No general-purpose training / fine-tuning hooks.** The framework is read-side only: you can inspect and modify activations during inference; you can't modify weights.
- **bf16 rounding non-determinism.** Running the same prompt twice can give bit-equal outputs in practice, but intermediate bf16 products don't promise it. Don't build tests that depend on bitwise equality across model loads.

---

## File map

```
mechbench_core/
├── __init__.py       # Public API re-exports
├── README.md         # This file
├── _arch.py          # Constants + hook-point registry (single source of truth)
├── _forward.py       # Canonical hook-aware forward pass
├── model.py          # Model.load / Model.run / RunResult
├── cache.py          # ActivationCache
├── hooks.py          # HookInfo + parse_hook_name
├── interventions.py  # Ablate / Capture / Patch / compose
├── lens.py           # logit_lens_final / logit_lens_per_position
├── geometry.py       # fact_vectors / centroid_decode / stats
├── plot.py           # 6 plot helpers
├── prompts.py        # Prompt / PromptSet / ValidatedPromptSet / validate()
├── errors.py         # InvalidHookName / LayerIndexOutOfRange / CacheKeyError
├── _smoke.py                  # Forward-path smoke test
├── _smoke_interventions.py    # Composition smoke test
└── _smoke_plots.py            # Plot-helpers smoke test
```

Project-specific prompt collections (FACTUAL_15, BIG_SWEEP_96, STRESS_*) live alongside the experiments that use them, in `experiments/prompts/`. The integration test that reproduces published findings against those collections is `experiments/smoke_analysis.py`.
