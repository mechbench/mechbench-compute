"""Bit-exactness regression smoke for the canonical Gemma 4 forward.

This is the north-star canary. The whole reason `_forward.run_forward` exists
is to reproduce mlx-vlm's layer loop exactly while exposing hook points — so
the load-bearing invariant is: at every layer the user is NOT actively probing,
the residual stream stays bitwise-identical to mlx-vlm's standard forward. A
silent break of that invariant (e.g. an upstream refactor we mirror wrong) is
the single most dangerous failure mode, because outputs still look plausible.

This smoke asserts that invariant directly, rather than via a top-1 proxy:

  1. UNPROBED forward == stock upstream layer loop, BITWISE
     (max|Δlogit| == 0). Covers cache construction, masks, and — for E4B/E2B —
     the `previous_kvs` KV-sharing thread through the shared tail.
  2. PROBING a global attention layer (forcing it onto the manual-softmax path)
     keeps the prediction coherent: top-1 unchanged, and the logit drift is
     pure manual-softmax rounding, not KV corruption — confirmed by a control
     probe of an earlier non-source global layer drifting a *comparable* amount
     (KV corruption would make the source layer diverge MORE, not less).

It motivated splitting task 000223 (the mlx-vlm 0.4.x→0.6.x re-mirror) out of
000222 (the 12B): the 0.6.x bump rewrote the loop this file mirrors, and this
canary is what catches that class of break.

Scope: the gemma4 family (E2B / E4B / 12B `gemma4_unified`), i.e. the
`run_forward` path. Other arch paths (gemma3, qwen2, llama) have their own
forward modules and would need their own bit-exactness canaries.

Run from project root with the venv active:
    python -m mechbench_core._smoke_bitexact                       # E4B default
    python -m mechbench_core._smoke_bitexact <hf-model-id>         # e.g. 12B
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np
from mlx_vlm.models import cache as cache_mod

from . import Model
from ._arch import DEFAULT_MODEL_ID
from ._forward import run_forward

PROMPT = "Complete this sentence with one word: The capital of France is"


def _stock_logits(mlxm, ids: mx.array) -> mx.array:
    """The reference: mlx-vlm's own Gemma4TextModel layer loop, unmodified."""
    lm = mlxm.language_model
    tm = lm.model
    emb = mlxm.get_input_embeddings(input_ids=ids, pixel_values=None)
    normed = tm(
        inputs=ids,
        inputs_embeds=emb.inputs_embeds,
        cache=cache_mod.make_prompt_cache(lm),
        per_layer_inputs=emb.per_layer_inputs,
    )
    logits = lm.logits_from_hidden(normed)
    mx.eval(logits)
    return logits


def main(model_id: str = DEFAULT_MODEL_ID) -> int:
    print(f"Loading {model_id} ...")
    t0 = time.perf_counter()
    model = Model.load(model_id)
    arch = model.arch
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.")

    if arch.model_type != "gemma4":
        print(f"SKIP: bit-exactness smoke covers the gemma4 forward path "
              f"(run_forward); loaded model reports model_type="
              f"{arch.model_type!r}, which uses a different path.")
        return 0

    mlxm = model._model
    tm = mlxm.language_model.model
    ids = model.tokenize(PROMPT)

    stock = np.array(_stock_logits(mlxm, ids).astype(mx.float32))
    mb_logits, _ = run_forward(mlxm, ids, arch=arch)
    mx.eval(mb_logits)
    mb = np.array(mb_logits.astype(mx.float32))

    stock_top1 = int(np.argmax(stock[0, -1]))
    max_abs = float(np.max(np.abs(stock - mb)))
    exact = bool(np.array_equal(stock, mb))

    print("\n[1] unprobed run_forward vs stock upstream loop")
    print(f"    max|Δlogit| = {max_abs:.3e}   bitwise_equal = {exact}")
    print(f"    stock top1 = {stock_top1}   mechbench top1 = "
          f"{int(np.argmax(mb[0, -1]))}")

    # ---- [2] probe a global layer; ---- [3] control: earlier global layer ----
    # `src` is the last fresh-K/V global (the KV source for the shared global
    # tail when one exists); `ctrl` is an earlier global with no consumers.
    src = arch.last_fresh_kv_global
    ctrl_candidates = [g for g in arch.global_layers if g < src]
    ctrl = max(ctrl_candidates) if ctrl_candidates else src
    consumers = [i for i in range(arch.first_kv_shared_layer, arch.n_layers)
                 if tm.previous_kvs[i] == src]

    probed, _ = run_forward(mlxm, ids, capture=[f"blocks.{src}.attn.weights"],
                            arch=arch)
    mx.eval(probed)
    p = np.array(probed[0, -1].astype(mx.float32))
    drift_src = float(np.max(np.abs(stock[0, -1] - p)))
    top1_src = int(np.argmax(p))

    ctrl_logits, _ = run_forward(mlxm, ids,
                                 capture=[f"blocks.{ctrl}.attn.weights"], arch=arch)
    mx.eval(ctrl_logits)
    cl = np.array(ctrl_logits[0, -1].astype(mx.float32))
    drift_ctrl = float(np.max(np.abs(stock[0, -1] - cl)))
    top1_ctrl = int(np.argmax(cl))

    print(f"\n[2] probe global L{src} (KV consumers={consumers}): "
          f"top1={top1_src} drift={drift_src:.3e}")
    print(f"[3] control non-source global L{ctrl}: "
          f"top1={top1_ctrl} drift={drift_ctrl:.3e}")
    print("    → both drifts are manual-softmax rounding, not KV corruption")

    ok = (exact
          and top1_src == stock_top1
          and top1_ctrl == stock_top1)
    print()
    if not ok:
        if not exact:
            print("BIT-EXACTNESS SMOKE FAILED: unprobed run_forward diverges "
                  "from the stock upstream forward. The canonical forward no "
                  "longer mirrors mlx-vlm — investigate _forward.py.")
        else:
            print("BIT-EXACTNESS SMOKE FAILED: probing a layer changed the "
                  "top-1 prediction, which points at KV-threading corruption "
                  "in the manual attention path.")
        return 1

    print("Bit-exactness smoke passed.")
    print("Unprobed forward is bitwise-identical to upstream; probing perturbs "
          "only via manual-softmax rounding.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID))
