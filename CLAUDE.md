# CLAUDE.md — mechbench-compute

The Python compute engine for the mechbench family. MLX-first, backend-agnostic in surface.

## Scope

This repo owns: model loading, hook-aware forward, interventions, activation cache, per-architecture adapters, logit lens, fact-vector geometry, probe primitives, head-weight static analysis, and matplotlib plot helpers.

This repo does NOT own:

- The typed emission schema (→ `mechbench-schema`).
- The TypeScript visualization layer (→ `mechbench-ui`).
- Agent tool-call surfaces (→ `mechbench-runner`).
- Project-specific research scripts, findings, essays (→ `mechbench-experiments`).
- Skill bundles / scaffolders (→ `mechbench-skills`).

If a user asks you to add code that belongs in one of those repos, push back. The split exists for a reason — see the meta repo's [PHILOSOPHY\_AND\_DIRECTION.md](https://github.com/mechbench/mechbench/blob/main/docs/PHILOSOPHY_AND_DIRECTION.md).

## Architecture notes

- The forward pass has one canonical path: `_forward.run_forward()`. Every experiment goes through `Model.run()`; nothing calls `model.language_model(input_ids)` directly.
- Per-architecture variation is captured in `_arch.Arch`. The hook-aware forwards cover Gemma 4, Gemma 3, Qwen 2.x and Llama 3.x.
- bf16 throughout the cache; float32 only at the analysis boundary (MLX → numpy conversions on bf16 arrays will crash with a PEP 3118 buffer format error).
- MLX is lazy; `Model.run` evals logits + cache in a single batch before returning.
- Manual vs fused attention path is selected automatically per-layer based on which hooks are registered; the residual stream stays bitwise-equivalent to mlx_vlm's standard forward at every layer the user isn't actively probing.

## Task tracking

Tasks for this repo are tracked outside it. Nothing in this repository names one: no task id in code, a string, a test or a doc (CHANGELOG.md aside). `tests/test_no_history.py` enforces it.

## Code conventions

- **No comments.** Only directives and `# external:` facts; see docs/COMMENTS.md.
- **No speculative abstractions.** Build for the second consumer, not the hypothetical tenth.
- **Read mlx-vlm source before guessing its API.** The upstream Gemma 4 model file is a few hundred lines of readable Python. When a forward-pass change surprises you, diff against the upstream first.
- **Prefer editing the canonical `_forward.py` over adding parallel forward paths.** The single-canonical-forward invariant is load-bearing; breaking it is how the original "garbage output" bug happened in the predecessor repo.
- **Smoke tests are the canary.** `python -m mechbench_compute._smoke` should always pass on `main`. Before landing a framework change, run it.

## Dev workflow

```bash
pip install -e '.[dev]'
python -m mechbench_compute._smoke
pytest tests/
ruff check .
```

## Session close

Standard: commit named files, push, verify remote. Work is not done until pushed.
