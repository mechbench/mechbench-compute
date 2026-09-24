from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np

from . import Model
from . import attribution as attr
from . import head_weights as hw

TWELVE_B_MODEL_ID = "mlx-community/gemma-4-12B-it-8bit"

PROMPTS_REQUIRE_TOP1 = [
    ("Complete this sentence with one word: The Eiffel Tower is in", "Paris"),
    ("Complete this sentence with one word: The capital of Japan is", "Tokyo"),
]

GLOBAL_LAYER = 23


def _check(label: str, got, expected) -> bool:
    ok = got == expected
    print(f"  {'[OK]' if ok else '[FAIL]'} {label}: {got!r}"
          + ("" if ok else f"  expected {expected!r}"))
    return ok


def main() -> int:
    print(f"Loading {TWELVE_B_MODEL_ID} ...")
    t0 = time.perf_counter()
    model = Model.load(TWELVE_B_MODEL_ID)
    a = model.arch
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.\n")

    ok = True

    print("Arch facts:")
    ok &= _check("model_type(family)", a.model_type, "gemma4")
    ok &= _check("n_layers", a.n_layers, 48)
    ok &= _check("d_model", a.d_model, 3840)
    ok &= _check("n_heads (sliding)", a.n_heads, 16)
    ok &= _check("n_kv_heads (sliding)", a.n_kv_heads, 8)
    ok &= _check("vocab_size", a.vocab_size, 262144)
    ok &= _check("global_layers", a.global_layers, (5, 11, 17, 23, 29, 35, 41, 47))
    ok &= _check("first_kv_shared_layer (no sharing)", a.first_kv_shared_layer, 48)

    print("\nForward (factual recall):")
    for prompt, expected in PROMPTS_REQUIRE_TOP1:
        ids = model.tokenize(prompt)
        r = model.run(ids)
        last = np.array(r.last_logits.astype(mx.float32))
        tok = model.tokenizer.decode([int(np.argmax(last))]).strip()
        ok &= _check(f"{expected} completion", tok, expected)

    print(f"\nGlobal-layer (L{GLOBAL_LAYER}) attention geometry via captured internals:")
    ids = model.tokenize(PROMPTS_REQUIRE_TOP1[0][0])
    seq = ids.shape[1]
    cap = [f"blocks.{GLOBAL_LAYER}.attn.k", f"blocks.{GLOBAL_LAYER}.attn.weights",
           f"blocks.{GLOBAL_LAYER}.attn.per_head_out"]
    r = model.run(ids, capture=cap)
    ok &= _check("attn.k shape [B,1,L,512]",
                 tuple(r.cache[cap[0]].shape), (1, 1, seq, 512))
    ok &= _check("attn.weights shape [B,16,L,L]",
                 tuple(r.cache[cap[1]].shape), (1, 16, seq, seq))
    ok &= _check("attn.per_head_out shape [B,16,L,512]",
                 tuple(r.cache[cap[2]].shape), (1, 16, seq, 512))

    print(f"\nStatic head-weight analysis at global L{GLOBAL_LAYER} (k_eq_v + quantized):")
    spec = hw.get_head_spec(model, GLOBAL_LAYER, head=0)
    ok &= _check("use_k_eq_v", spec.use_k_eq_v, True)
    ok &= _check("head_dim", spec.head_dim, 512)
    ok &= _check("n_kv_heads", spec.n_kv_heads, 1)
    ok &= _check("kv_group (MQA → 0)", spec.kv_group, 0)
    ok &= _check("W_V dequantized shape [head_dim, d_model]",
                 tuple(spec.W_V.shape), (512, 3840))
    qk = hw.qk_circuit(model, GLOBAL_LAYER, 0)
    ov = hw.ov_circuit(model, GLOBAL_LAYER, 0)
    qk_ok = len(qk.components) > 0 and np.isfinite(qk.components[0].strength)
    ov_ok = len(ov.components) > 0 and np.isfinite(ov.components[0].strength)
    ok &= _check("qk_circuit finite σ", qk_ok, True)
    ok &= _check("ov_circuit finite σ", ov_ok, True)

    r = model.run(ids, capture=[f"blocks.{GLOBAL_LAYER}.attn.per_head_out"])
    hr = attr.head_results(model, r.cache, layer=GLOBAL_LAYER)
    ok &= _check("head_results shape [16,L,3840]", tuple(hr.shape), (16, seq, 3840))

    print()
    if not ok:
        print("12B VERIFICATION SMOKE FAILED.")
        return 1
    print("12B verification smoke test passed.")
    print("Forward, captured global-layer geometry (1 KV head / head_dim 512), "
          "and k_eq_v + quantized static head analysis all work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
