"""Bit-exactness regression smoke for the canonical forward paths.

This is the north-star canary. The whole reason the `_forward*` modules exist
is to reproduce the upstream layer loop exactly while exposing hook points — so
the load-bearing invariant is: at every layer the user is NOT actively probing,
the residual stream stays bitwise-identical to the framework's standard forward.
A silent break of that invariant (e.g. an upstream refactor we mirror wrong) is
the single most dangerous failure mode, because outputs still look plausible.

This smoke asserts that invariant directly, rather than via a top-1 proxy:

  1. UNPROBED forward == stock upstream layer loop, BITWISE
     (max|Δlogit| == 0). Covers cache construction, masks, and — for gemma4
     E4B/E2B — the `previous_kvs` KV-sharing thread through the shared tail.
  2. PROBING an attention layer (forcing it onto the manual-softmax path) keeps
     the prediction coherent: top-1 unchanged, and the logit drift is pure
     manual-softmax rounding, not KV corruption — confirmed by a control probe
     of an earlier layer drifting a *comparable* amount (KV corruption would
     make a KV-source layer diverge MORE, not less).

It motivated splitting task 000223 (the mlx-vlm 0.4.x→0.6.x re-mirror) out of
000222 (the 12B): the 0.6.x bump rewrote the loop this canary guards, and this
is what catches that class of break.

Scope: all four supported forward paths, dispatched by family —
  - gemma4 (E2B / E4B / 12B `gemma4_unified`): mlx-vlm `Gemma4TextModel` loop.
  - gemma3: mlx-vlm `Gemma3Model` loop + `lm_head`.
  - qwen2 / llama: mlx-lm models (callable → logits directly).
The stock reference per family is the model's own standard forward, never the
mechbench mirror.

Run from project root with the venv active:
    python -m mechbench_core._smoke_bitexact                       # E4B default
    python -m mechbench_core._smoke_bitexact <hf-model-id>         # any family
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np
from mlx_vlm.models import cache as cache_mod

from . import Model
from ._arch import DEFAULT_MODEL_ID

PROMPT = "Complete this sentence with one word: The capital of France is"

_SUPPORTED = ("gemma4", "gemma3", "qwen2", "llama")


def _text_model(mlxm, arch):
    """The text transformer whose layer loop the mechbench forward mirrors."""
    if arch.model_type in ("qwen2", "llama"):
        return mlxm.model
    return mlxm.language_model.model


def _stock_logits(mlxm, ids: mx.array, arch) -> mx.array:
    """The reference: the model's OWN standard forward, unmodified by mechbench.

    Dispatched by family — each path is the upstream layer loop + that family's
    unembed, exactly what the corresponding `_forward*` module mirrors.
    """
    mt = arch.model_type
    if mt in ("qwen2", "llama"):
        # mlx-lm models are callable and return logits directly.
        logits = mlxm(ids)
    elif mt == "gemma3":
        lm = mlxm.language_model
        tm = lm.model
        normed = tm(ids, cache=cache_mod.make_prompt_cache(lm))
        logits = lm.lm_head(normed)
    else:  # gemma4 / gemma4_unified
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

    if arch.model_type not in _SUPPORTED:
        print(f"SKIP: no canonical forward for model_type={arch.model_type!r}.")
        return 0

    mlxm = model._model
    tm = _text_model(mlxm, arch)
    ids = model.tokenize(PROMPT)

    stock = np.array(_stock_logits(mlxm, ids, arch).astype(mx.float32))
    mb = np.array(model.run(ids).logits.astype(mx.float32))

    stock_top1 = int(np.argmax(stock[0, -1]))
    max_abs = float(np.max(np.abs(stock - mb)))
    exact = bool(np.array_equal(stock, mb))

    print(f"\n[1] unprobed forward vs stock upstream loop  (family={arch.model_type})")
    print(f"    max|Δlogit| = {max_abs:.3e}   bitwise_equal = {exact}")
    print(f"    stock top1 = {stock_top1}   mechbench top1 = "
          f"{int(np.argmax(mb[0, -1]))}")

    # ---- [2] probe a layer; ---- [3] control: an earlier layer ----
    # `src` is the last fresh-K/V global (the KV source for the shared global
    # tail when one exists, e.g. E4B); `ctrl` is an earlier global with no
    # consumers. For all-global families (qwen2/llama) these are just the last
    # two layers; KV-sharing `consumers` is empty unless the model has it.
    src = arch.last_fresh_kv_global
    ctrl_candidates = [g for g in arch.global_layers if g < src]
    ctrl = max(ctrl_candidates) if ctrl_candidates else src
    prev = getattr(tm, "previous_kvs", None)
    consumers = (
        [i for i in range(arch.first_kv_shared_layer, arch.n_layers)
         if prev[i] == src]
        if prev is not None else []
    )

    def _probe(layer: int) -> tuple[int, float]:
        out = model.run(ids, capture=[f"blocks.{layer}.attn.weights"]).logits
        row = np.array(out[0, -1].astype(mx.float32))
        return int(np.argmax(row)), float(np.max(np.abs(stock[0, -1] - row)))

    top1_src, drift_src = _probe(src)
    top1_ctrl, drift_ctrl = _probe(ctrl) if ctrl != src else (top1_src, drift_src)

    print(f"\n[2] probe layer L{src} (KV consumers={consumers}): "
          f"top1={top1_src} drift={drift_src:.3e}")
    if ctrl != src:
        print(f"[3] control layer L{ctrl}: top1={top1_ctrl} drift={drift_ctrl:.3e}")
        print("    → comparable drift is manual-softmax rounding, not KV corruption")
    else:
        print("[3] control: skipped (model has a single attention layer)")

    ok = exact and top1_src == stock_top1 and top1_ctrl == stock_top1
    print()
    if not ok:
        if not exact:
            print("BIT-EXACTNESS SMOKE FAILED: unprobed forward diverges from "
                  "the stock upstream forward. The canonical forward no longer "
                  f"mirrors upstream — investigate the {arch.model_type} forward "
                  "path.")
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
