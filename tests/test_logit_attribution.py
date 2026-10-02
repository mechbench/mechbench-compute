from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import attribution
from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.distill import render
from mechbench_compute.ops.logits.attribute import attribute_logits
from tests.tiny_models import build_tiny_model

SCALARS = (0.5, 0.25, 2.0, 0.8)

RECORD = {"id": "r", "text": "the cat sat on a", "tracked": {"answer": "mat"}}


def build_scaled_gemma4():
    model = build_tiny_model("gemma4", BY_MODEL_TYPE["gemma4"])
    for layer, s in zip(model.lm.model.layers, SCALARS, strict=True):
        layer.layer_scalar = mx.array([s])
    return model


def read_last(cache, name: str) -> np.ndarray:
    return np.array(mx.array(cache[name]).astype(mx.float32), dtype=np.float64)[0, -1:]


def test_under_layer_scalars_a_layer_is_its_writes_times_the_scalars_from_it_on():
    model = build_scaled_gemma4()
    row = attribute_logits(model, [RECORD], {})["items"][0]
    tok = row["target"]["id"]
    n, writes = len(SCALARS), ("attn_out", "mlp_out", "gate_out")
    result = model.run(render(model, RECORD).array, capture=[
        *(f"blocks.{i}.{p}" for i in range(n) for p in (*writes, "resid_post")),
        "blocks.0.resid_pre", "final_norm.scale"])
    cache = result.cache
    ln_scale = np.array(mx.array(cache["final_norm.scale"]).astype(mx.float32)).reshape(-1)

    def project(stack: np.ndarray) -> np.ndarray:
        return attribution.logit_attrs(model, stack, [tok], apply_ln=True, ln_scale=ln_scale)[:, 0]

    own = np.stack([read_last(cache, "blocks.0.resid_pre") * math.prod(SCALARS)]
                   + [sum(read_last(cache, f"blocks.{i}.{w}") for w in writes) * math.prod(SCALARS[i:])
                      for i in range(n)])
    want = project(own)
    assert np.allclose(row["measures"]["contribution"], want, atol=1e-3)
    assert row["additivity"]["residual"] == pytest.approx(0.0, abs=2e-3)

    stream = attribution.accumulated_resid(cache, include_pre=True)
    diffed = project(np.diff(stream, axis=0, prepend=np.zeros_like(stream[:1])))
    assert diffed.sum() == pytest.approx(want.sum(), abs=1e-3)
    assert not np.allclose(diffed[1:], want[1:], atol=1e-2)


@pytest.mark.parametrize("name,gate", [("gemma4", True), ("gemma4-31b", False)])
def test_by_sublayer_a_layer_has_the_writes_its_checkpoint_has(name, gate):
    model = build_tiny_model(name, BY_MODEL_TYPE["gemma4"])
    by_layer = attribute_logits(model, [RECORD], {})
    by_sublayer = attribute_logits(model, [RECORD], {"split": "sublayer"})
    n = model.arch.n_layers
    assert (by_layer["split"], by_sublayer["split"]) == ("layer", "sublayer")
    assert by_layer["components"] == ["embed", *(f"L{i}" for i in range(n))]
    writes = ("attn", "mlp", "gate") if gate else ("attn", "mlp")
    assert by_sublayer["components"] == ["embed", *(f"L{i}.{w}" for i in range(n) for w in writes)]
    layer = np.array(by_layer["items"][0]["measures"]["contribution"])
    pieces = np.array(by_sublayer["items"][0]["measures"]["contribution"])
    assert pieces[0] == pytest.approx(layer[0], abs=1e-3)
    assert np.allclose(pieces[1:].reshape(n, len(writes)).sum(axis=1), layer[1:], atol=1e-3)


@pytest.mark.parametrize("build", [build_scaled_gemma4,
                                   lambda: build_tiny_model("gemma3", BY_MODEL_TYPE["gemma3"]),
                                   lambda: build_tiny_model("qwen2", BY_MODEL_TYPE["qwen2"])],
                         ids=["gemma4-scaled", "gemma3", "qwen2"])
def test_a_layer_s_heads_sum_to_its_attention_piece(build):
    model = build()
    n = model.arch.n_layers
    out = attribute_logits(model, [RECORD], {"split": "sublayer", "per_head_layers": list(range(n))})
    row = out["items"][0]
    attn = dict(zip(out["components"], row["measures"]["contribution"]))
    assert [h["layer"] for h in row["per_head"]] == list(range(n))
    for heads in row["per_head"]:
        assert len(heads["contributions"]) == model.arch.n_heads and "bias" not in heads
        assert sum(heads["contributions"]) == pytest.approx(attn[f"L{heads['layer']}.attn"], abs=1e-3)


def test_an_unknown_split_is_refused_by_name():
    model = build_tiny_model("llama", BY_MODEL_TYPE["llama"])
    with pytest.raises(ValueError, match="layer or sublayer, not 'head'"):
        attribute_logits(model, [RECORD], {"split": "head"})
