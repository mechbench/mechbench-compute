"""Smoke test for attribution.py: accumulated_resid, decompose_resid,
head_results, logit_attrs.

Runs a single forward pass with enough captures to exercise every code
path, then asserts the outputs have the right shapes and sane values.
"""

from __future__ import annotations

import numpy as np

from ._arch import D_MODEL, N_LAYERS
from .attribution import (
    accumulated_resid,
    decompose_resid,
    head_results,
    logit_attrs,
)
from .interventions import Capture
from .model import Model


def main() -> None:
    print("Loading model...")
    model = Model.load()
    n_heads = model.arch.n_heads

    prompt = "Complete this sentence with one word: The capital of France is"
    ids = model.tokenize(prompt)
    seq_len = int(ids.shape[1])
    print(f"Prompt tokenized: seq_len={seq_len}")

    interventions = [
        Capture.residual(range(N_LAYERS), point="post"),
        Capture.residual([0], point="pre"),
        Capture.attn_out(range(N_LAYERS)),
        Capture.mlp_out(range(N_LAYERS)),
        Capture.gate_out(range(N_LAYERS)),
        Capture.per_head_out([14, 23, 29]),
    ]
    print("Running forward pass with attribution captures...")
    result = model.run(ids, interventions=interventions)
    cache = result.cache

    # --- accumulated_resid ---
    stack = accumulated_resid(cache)
    assert stack.shape == (N_LAYERS, seq_len, D_MODEL), stack.shape
    assert stack.dtype == np.float32
    print(f"accumulated_resid:          {stack.shape}  dtype={stack.dtype}")

    stack_with_pre = accumulated_resid(cache, include_pre=True)
    assert stack_with_pre.shape == (N_LAYERS + 1, seq_len, D_MODEL)
    print(f"accumulated_resid(include_pre): {stack_with_pre.shape}")

    # --- decompose_resid ---
    parts = decompose_resid(cache)
    for k in ("attn", "mlp", "gate"):
        assert parts[k].shape == (N_LAYERS, seq_len, D_MODEL), (k, parts[k].shape)
    print(
        f"decompose_resid:            attn={parts['attn'].shape} "
        f"mlp={parts['mlp'].shape} gate={parts['gate'].shape}"
    )

    # --- head_results ---
    heads23 = head_results(model, cache, layer=23)
    assert heads23.shape == (n_heads, seq_len, D_MODEL), heads23.shape
    print(f"head_results(layer=23):     {heads23.shape}")

    # --- logit_attrs ---
    paris_id = int(model.tokenizer.encode(" Paris", add_special_tokens=False)[0])
    berlin_id = int(model.tokenizer.encode(" Berlin", add_special_tokens=False)[0])
    print(f"Target tokens: Paris={paris_id}, Berlin={berlin_id}")

    layer_attrs = logit_attrs(model, stack, [paris_id, berlin_id])
    assert layer_attrs.shape == (N_LAYERS, 2), layer_attrs.shape
    print(f"logit_attrs(layers):        {layer_attrs.shape}")

    head_attrs = logit_attrs(model, heads23, [paris_id, berlin_id])
    assert head_attrs.shape == (n_heads, 2), head_attrs.shape
    print(f"logit_attrs(heads at L23):  {head_attrs.shape}")

    # Sanity: by the final layer, Paris attribution should beat Berlin for
    # a France-capital prompt (directionally — we don't assert a magnitude).
    final_paris = float(layer_attrs[-1, 0])
    final_berlin = float(layer_attrs[-1, 1])
    print(
        f"  Final-layer attribution:  Paris={final_paris:+.3f}  "
        f"Berlin={final_berlin:+.3f}  diff={final_paris - final_berlin:+.3f}"
    )
    assert final_paris > final_berlin, (
        "By the final layer, Paris should outscore Berlin on 'capital of France'. "
        f"Got Paris={final_paris}, Berlin={final_berlin}"
    )

    print("\nAttribution smoke test passed.")


if __name__ == "__main__":
    main()
