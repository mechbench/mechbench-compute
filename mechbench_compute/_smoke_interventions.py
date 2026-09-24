from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np

from . import Ablate, Capture, Model


def main() -> int:
    print("Loading model...")
    t0 = time.perf_counter()
    model = Model.load("mlx-community/gemma-4-E4B-it-bf16")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.\n")

    ids = model.tokenize(
        "Complete this sentence with one word: The Eiffel Tower is in"
    )

    result = model.run(ids, interventions=[
        Ablate.head(29, head=7),
        Capture.per_head_out([29]),
    ])
    captured = np.array(
        result.cache["blocks.29.attn.per_head_out"].astype(mx.float32)
    )
    head7_zero = bool(np.all(captured[:, 7, :, :] == 0))
    other_heads_nonzero = bool(np.any(captured[:, 0, :, :] != 0))
    ok = head7_zero and other_heads_nonzero
    print(
        f"  [{'OK' if ok else 'FAIL'}] "
        f"Composition: Ablate.head(29, head=7) + Capture.per_head_out([29])\n"
        f"        head7_all_zero={head7_zero}  "
        f"other_heads_have_signal={other_heads_nonzero}"
    )

    if not ok:
        print("\nINTERVENTION-COMPOSITION SMOKE TEST FAILED.")
        return 1
    print("\nIntervention-composition smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
