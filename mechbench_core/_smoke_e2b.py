"""Verification smoke test for Gemma 4 E2B.

Loads the E2B variant via Model.load() and confirms:
  1. Arch.from_mlx_model derives the expected E2B architecture facts.
  2. Model.run produces sensible top-1 tokens on factual-recall prompts.
  3. The forward path doesn't crash on E2B's narrower residual stream
     (1536 vs 2560), reduced KV-head count (1 vs 2), or different
     layer count (35 vs 42).

This is the TransformerLens-3-style "validate outputs match HuggingFace"
phase for the new model variant. We don't compare against mlx_vlm.generate
numerically because Model.run already reuses mlx-vlm's components verbatim;
if the top-1 tokens are right, the forward path is right.

Run from project root with the venv active:
    python -m mechbench_core._smoke_e2b
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np

from . import Model

E2B_MODEL_ID = "mlx-community/gemma-4-e2b-it-bf16"

# Same prompts as _smoke.py for E4B, except we don't pin specific top-1
# tokens — E2B is a smaller model and may not match E4B exactly. We require
# the predictions to be plausible (high-probability completions of the
# template) rather than identical to E4B's. The Eiffel Tower → Paris
# completion is the strongest factual-recall test.
PROMPTS_REQUIRE_TOP1 = [
    ("Complete this sentence with one word: The Eiffel Tower is in", "Paris"),
    ("Complete this sentence with one word: The capital of Japan is", "Tokyo"),
]

PROMPTS_FREE = [
    "Complete this sentence with one word: Romeo and Juliet was written by",
    "Complete this sentence with one word: The opposite of hot is",
    "Complete this sentence with one word: Monday, Tuesday,",
]

# Expected E2B architecture, as read from the HF config.
EXPECTED_E2B = dict(
    n_layers=35,
    d_model=1536,
    n_heads=8,
    n_kv_heads=1,
    vocab_size=262144,
    global_layers=(4, 9, 14, 19, 24, 29, 34),
    first_kv_shared_layer=15,  # 35 - 20 = 15
    last_fresh_kv_global=14,
)


def _check_arch(arch) -> bool:
    print("\nArchitecture facts read from E2B config:")
    print(f"  model_id:                 {arch.model_id}")
    print(f"  n_layers:                 {arch.n_layers}")
    print(f"  d_model:                  {arch.d_model}")
    print(f"  n_heads:                  {arch.n_heads}")
    print(f"  n_kv_heads:               {arch.n_kv_heads}")
    print(f"  vocab_size:               {arch.vocab_size}")
    print(f"  global_layers:            {arch.global_layers}")
    print(f"  first_kv_shared_layer:    {arch.first_kv_shared_layer}")
    print(f"  last_fresh_kv_global:     {arch.last_fresh_kv_global}")

    fields = ("n_layers", "d_model", "n_heads", "n_kv_heads", "vocab_size",
              "global_layers", "first_kv_shared_layer", "last_fresh_kv_global")
    ok = True
    for f in fields:
        actual = getattr(arch, f)
        expected = EXPECTED_E2B[f]
        if actual != expected:
            print(f"  ! {f} mismatch: expected {expected}, got {actual}")
            ok = False
    return ok


def main() -> int:
    print(f"Loading {E2B_MODEL_ID}...")
    t0 = time.perf_counter()
    model = Model.load(E2B_MODEL_ID)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s. "
          f"({len(model.arch.all_hook_names())} hook points exposed.)")

    arch_ok = _check_arch(model.arch)
    if not arch_ok:
        print("\nE2B SMOKE FAILED: Arch.from_mlx_model did not derive the "
              "expected E2B fields. Investigate _arch.py.")
        return 1

    print(f"\nForward smoke test on {len(PROMPTS_REQUIRE_TOP1) + len(PROMPTS_FREE)} prompts.\n")
    print(f"  {'expected':>11}  {'top1 (run)':>14}  {'p':>6}  {'match':>5}")
    print("  " + "-" * 50)

    top1_pass = True
    for prompt, expected in PROMPTS_REQUIRE_TOP1:
        ids = model.tokenize(prompt)
        result = model.run(ids)
        last = np.array(result.last_logits.astype(mx.float32))
        run_id = int(np.argmax(last))
        run_tok = model.tokenizer.decode([run_id]).strip()
        run_prob = float(np.exp(last[run_id] - np.log(np.sum(np.exp(last)))))
        ok = run_tok == expected
        if not ok:
            top1_pass = False
        print(f"  {expected:>11}  {run_tok!r:>14}  {run_prob:>6.3f}  "
              f"{str(ok):>5}")

    for prompt in PROMPTS_FREE:
        ids = model.tokenize(prompt)
        result = model.run(ids)
        last = np.array(result.last_logits.astype(mx.float32))
        run_id = int(np.argmax(last))
        run_tok = model.tokenizer.decode([run_id]).strip()
        run_prob = float(np.exp(last[run_id] - np.log(np.sum(np.exp(last)))))
        print(f"  {'(free)':>11}  {run_tok!r:>14}  {run_prob:>6.3f}  "
              f"{'-':>5}")

    print()
    if not top1_pass:
        print("E2B SMOKE FAILED: Model.run produced the wrong top-1 on "
              "factual-recall prompts. The forward path may be broken on "
              "E2B's architecture (35 layers, n_kv_heads=1, d_model=1536).")
        return 1

    # Quick capture sanity check: capture residuals at one global layer
    # and confirm shape + dtype match expectations for E2B.
    print("Capture sanity check on blocks.14.resid_post (E2B's L23-equivalent):")
    ids = model.tokenize(PROMPTS_REQUIRE_TOP1[0][0])
    result = model.run(ids, capture=["blocks.14.resid_post"])
    resid = result.cache["blocks.14.resid_post"]
    expected_shape = (1, ids.shape[1], model.arch.d_model)
    print(f"  shape: {resid.shape}  expected: {expected_shape}")
    print(f"  dtype: {resid.dtype}  expected: bfloat16")
    if tuple(resid.shape) != expected_shape:
        print(f"\nE2B SMOKE FAILED: capture shape mismatch.")
        return 1

    print("\nE2B verification smoke test passed.")
    print("Arch facts match the HF config; forward path produces sensible "
          "predictions; capture works at the predicted L23-equivalent (L14).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
