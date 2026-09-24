from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np

from . import Model, all_hook_names

PROMPTS = [
    ("Complete this sentence with one word: The Eiffel Tower is in", "Paris"),
    ("Complete this sentence with one word: The capital of Japan is", "Tokyo"),
    ("Complete this sentence with one word: Romeo and Juliet was written by",
     "Shakespeare"),
    ("Complete this sentence with one word: The opposite of hot is", "cold"),
    ("Complete this sentence with one word: Monday, Tuesday,", "Wednesday"),
]


def main() -> int:
    print("Loading model...")
    t0 = time.perf_counter()
    model = Model.load("mlx-community/gemma-4-E4B-it-bf16")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s. "
          f"({len(all_hook_names())} hook points exposed.)")

    print(f"\nForward smoke test on {len(PROMPTS)} prompts.\n")
    print(f"  {'expected':>11}  {'top1 (run)':>14}  {'p':>6}  {'match':>5}")
    print("  " + "-" * 50)

    all_pass = True
    for prompt, expected in PROMPTS:
        ids = model.tokenize(prompt)
        result = model.run(ids)
        last = np.array(result.last_logits.astype(mx.float32))
        run_id = int(np.argmax(last))
        run_tok = model.tokenizer.decode([run_id]).strip()
        run_prob = float(np.exp(last[run_id] - np.log(np.sum(np.exp(last)))))
        ok = run_tok == expected
        if not ok:
            all_pass = False
        print(f"  {expected:>11}  {run_tok!r:>14}  {run_prob:>6.3f}  "
              f"{str(ok):>5}")

    print()
    if not all_pass:
        print("FORWARD SMOKE TEST FAILED: Model.run produced the wrong token on some prompt.")
        print("Investigate _forward.py — the canonical forward path is broken.")
        return 1

    print("Forward smoke test passed.")
    print("Model.run produces the expected top-1 token on all prompts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
