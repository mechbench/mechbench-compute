from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import CELLS, F, TOKEN, VARIANTS

KIND = Kind(
    "logits/attribution",
    "A record's direct logit attribution: each component's contribution to the target logit, with the additivity check that says the pieces sum to the truth.",
    extends="activations/grid",
    fields={"target": TOKEN, "contrast": TOKEN, "variants": VARIANTS, "contrast_variants": VARIANTS,
            "per_head": F("array", "`{layer, contributions[head]}` for the layers split by head, with `bias` where "
                          "`o_proj` has one: the layer's attention piece, which its heads and `bias` sum to.",
                          items={"type": "object"}),
            "additivity": F("object", "`{summed, true_logit, residual}` — the honesty number."),
            "features": F("array", "With a dictionary: `{index, activation, dla, contribution}` for the features "
                          "that contribute most, each its activation times its decoder row's direct logit "
                          "attribution.", items={"type": "object"}),
            "reconstruction": F("object", "With a whole dictionary: `{stream, features, b_dec, error}`, the "
                                "attribution of the activation the dictionary reads, split into the features' "
                                "sum, the decoder bias and what the dictionary does not reconstruct."),
            "cells": CELLS({"contribution": "The component's contribution to the target logit."}, component=True)},
    required=("id", "axes", "measures", "target", "additivity"),
    key=("id",),
    header={"components": "The component names: `embed`, then `L0`, `L1`, … by layer, or `L0.attn`, `L0.mlp` "
                          "(and `L0.gate`), `L1.attn`, … by sublayer.",
            "split": "`layer` or `sublayer`: what a component is.",
            "apply_ln": "Whether the final norm was folded in.",
            "layers": "The layers decomposed (all of them).",
            "dictionary": "The dictionary attributed to: `{kind, hash, derivation, reads, width, source}`.",
            "feature": "The one feature attributed to: `{dictionary, index}`."},
    doc="Axis `[component]`, in the header's `components` order; measure `contribution`. `records/unnest field: cells` makes one record per component, `{address, point, layer, position, component, contribution}` at position -1: `embed` is the residual entering the first layer, `L{first}.resid_pre@-1`; `L3` is layer 3's whole write to the stream, point `block`, `L3.block@-1`; and `L3.attn`, `L3.mlp` and `L3.gate` are its writes at `attn_out`, `mlp_out` and `gate_out`, `L3.attn_out@-1` and so on.",
)
