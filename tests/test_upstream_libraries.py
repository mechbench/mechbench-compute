from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mechbench_compute import (
    _forward,
    _forward_gemma3,
    _forward_llama,
    _forward_qwen,
    attribution,
    head_weights,
    peft,
)
from mechbench_compute._arch import Arch
from mechbench_compute.cache import ActivationCache


class _Passthrough:
    def __call__(self, x, offset=0):
        return x


def _identity_linear(d):
    lin = nn.Linear(d, d, bias=False)
    lin.weight = mx.eye(d)
    return lin


@pytest.mark.parametrize("forward", [_forward, _forward_gemma3, _forward_llama, _forward_qwen])
def test_the_manual_attention_path_makes_a_causal_string_mask_causal(forward):
    d, n = 4, 5
    attn = SimpleNamespace(
        n_heads=1, n_kv_heads=1, head_dim=d, scale=1.0, use_k_eq_v=False,
        q_proj=_identity_linear(d), k_proj=_identity_linear(d),
        v_proj=_identity_linear(d), o_proj=_Passthrough(),
        q_norm=_Passthrough(), k_norm=_Passthrough(), v_norm=_Passthrough(),
        rope=_Passthrough(),
    )
    cache = ActivationCache()
    x = mx.random.normal((1, n, d), key=mx.random.key(0))
    extra = {"shared_kv": None, "offset": 0} if forward is _forward else {}
    forward._attention_with_internals(
        SimpleNamespace(self_attn=attn), x, "causal", None, hooks={},
        capture_set={"blocks.0.attn.weights"}, cache=cache, layer_idx=0, **extra)
    weights = np.array(cache["blocks.0.attn.weights"])[0, 0]
    assert np.allclose(np.triu(weights, k=1), 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def _quantized(out_dim, in_dim, seed):
    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.weight = mx.random.normal((out_dim, in_dim), key=mx.random.key(seed)) * 0.1
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8), np.array(lin.weight)


def test_a_quantized_k_eq_v_layer_reads_dense_weights_and_values_from_k_proj():
    d, head_dim = 64, 64
    q, q_ref = _quantized(2 * head_dim, d, 1)
    k, k_ref = _quantized(head_dim, d, 2)
    o, o_ref = _quantized(d, 2 * head_dim, 3)
    attn = SimpleNamespace(
        head_dim=head_dim, n_heads=2, n_kv_heads=1, use_k_eq_v=True,
        q_proj=q, k_proj=k, o_proj=o, layer_type="full_attention",
        is_kv_shared_layer=False)
    model = SimpleNamespace(_model=SimpleNamespace(language_model=SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]))))
    spec = head_weights.get_head_spec(model, 0, 1)
    assert spec.W_Q.shape == (head_dim, d) and spec.W_O.shape == (d, head_dim)
    assert np.allclose(spec.W_Q, q_ref[head_dim:], atol=0.01)
    assert np.allclose(spec.W_K, k_ref, atol=0.01)
    assert np.array_equal(spec.W_V, spec.W_K)
    assert np.allclose(spec.W_O, o_ref[:, head_dim:], atol=0.01)

    per_head = mx.random.normal((1, 2, 3, head_dim), key=mx.random.key(4))
    cache = ActivationCache({"blocks.0.attn.per_head_out": per_head})
    out = attribution.head_results(model, cache, 0)
    want = np.array(per_head)[0, 1] @ o_ref[:, head_dim:].T
    assert out.shape == (2, 3, d) and np.allclose(out[1], want, atol=0.05)


def _norm_gain_model(model_type, norm):
    inner = SimpleNamespace(norm=norm)
    if model_type in ("qwen2", "llama"):
        wrapped = SimpleNamespace(model=inner)
    else:
        wrapped = SimpleNamespace(language_model=SimpleNamespace(model=inner))
    return SimpleNamespace(arch=SimpleNamespace(model_type=model_type), _model=wrapped)


@pytest.mark.parametrize("model_type", ["gemma3", "gemma4", "llama", "qwen2"])
def test_the_final_norm_gain_is_what_the_family_s_own_norm_multiplies_by(model_type):
    from mlx_lm.models import llama, qwen2
    from mlx_vlm.models.gemma3 import language as gemma3
    from mlx_vlm.models.gemma4 import language as gemma4

    make = {"gemma3": lambda d: gemma3.RMSNorm(d, eps=1e-6),
            "gemma4": lambda d: gemma4.Gemma4TextModel(gemma4.TextConfig(
                hidden_size=d, num_hidden_layers=1, intermediate_size=d,
                num_attention_heads=1, num_key_value_heads=1, head_dim=d,
                global_head_dim=d, vocab_size=8, vocab_size_per_layer_input=8,
                hidden_size_per_layer_input=0, num_kv_shared_layers=0)).norm,
            "llama": lambda d: llama.nn.RMSNorm(d, eps=1e-6),
            "qwen2": lambda d: qwen2.nn.RMSNorm(d, eps=1e-6)}[model_type]
    d = 16
    norm = make(d)
    norm.weight = mx.random.normal((d,), key=mx.random.key(5)) * 0.3
    x = np.array(mx.random.normal((d,), key=mx.random.key(6)))
    rms = np.sqrt((x * x).mean() + 1e-6)
    gain = attribution._final_norm_gain(_norm_gain_model(model_type, norm))
    assert np.allclose(x / rms * gain, np.array(norm(mx.array(x))), atol=1e-4)


def _vlm_model(text_config):
    return SimpleNamespace(config=SimpleNamespace(text_config=text_config))


def test_gemma4_globals_come_from_layer_types_and_the_shared_tail_from_its_count():
    from mlx_vlm.models.gemma4.config import TextConfig

    layer_types = ["full_attention" if i % 6 == 5 else "sliding_attention"
                   for i in range(42)]
    cfg = TextConfig.from_dict({"model_type": "gemma4_text", "num_hidden_layers": 42,
                                "hidden_size": 2560, "num_attention_heads": 8,
                                "num_key_value_heads": 2, "vocab_size": 262144,
                                "hidden_size_per_layer_input": 256,
                                "num_kv_shared_layers": 18, "layer_types": layer_types})
    arch = Arch.from_mlx_model(_vlm_model(cfg))
    assert arch.model_type == "gemma4" and arch.n_layers == 42
    assert arch.global_layers == (5, 11, 17, 23, 29, 35, 41)
    assert arch.first_kv_shared_layer == 24 and arch.last_fresh_kv_global == 23


def test_gemma3_globals_close_each_sliding_window_group_and_nothing_is_shared():
    from mlx_vlm.models.gemma3.config import TextConfig

    cfg = TextConfig.from_dict({"model_type": "gemma3_text", "num_hidden_layers": 34,
                                "hidden_size": 2560, "intermediate_size": 10240})
    arch = Arch.from_mlx_model(_vlm_model(cfg))
    assert arch.model_type == "gemma3"
    assert arch.global_layers == (5, 11, 17, 23, 29)
    assert arch.first_kv_shared_layer == 34


@pytest.mark.parametrize(("model_type", "layer_types", "globals_"), [
    ("qwen2", None, (0, 1, 2, 3)),
    ("llama", None, (0, 1, 2, 3)),
    ("llama", ["sliding_attention", "full_attention"] * 2, (1, 3)),
])
def test_an_mlx_lm_model_is_all_global_unless_it_declares_layer_types(
        model_type, layer_types, globals_):
    args = SimpleNamespace(model_type=model_type, num_hidden_layers=4, hidden_size=8,
                           num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
                           layer_types=layer_types)
    arch = Arch.from_mlx_model(SimpleNamespace(args=args))
    assert arch.global_layers == globals_ and arch.first_kv_shared_layer == 4


def _adapter(tmp_path, rank=2, d_in=6, d_out=4):
    a = mx.random.normal((rank, d_in), key=mx.random.key(7))
    b = mx.random.normal((d_out, rank), key=mx.random.key(8))
    path = tmp_path / "ours.safetensors"
    mx.save_safetensors(str(path), {"model.layers.0.self_attn.q_proj.lora_a": a,
                                    "model.layers.0.self_attn.q_proj.lora_b": b})
    return {"kind": "adapter/lora", "data": path.read_bytes(), "base_model": "org/base",
            "lora": {"rank": rank, "alpha": 8.0, "target_modules": ["q_proj"]}}, a, b


def test_an_adapter_leaves_and_returns_under_peft_s_own_names(tmp_path):
    adapter, a, b = _adapter(tmp_path)
    out = tmp_path / "peft"
    peft.peft_export(adapter, str(out))
    weights = dict(mx.load(str(out / "adapter_model.safetensors")))
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    assert set(weights) == {f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"}
    assert weights[f"{prefix}.lora_A.weight"].shape == (2, 6)
    assert weights[f"{prefix}.lora_B.weight"].shape == (4, 2)
    config = json.loads((out / "adapter_config.json").read_text())
    assert {k: config[k] for k in ("peft_type", "task_type", "r", "lora_alpha",
                                   "target_modules", "base_model_name_or_path")} == {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2, "lora_alpha": 8.0,
        "target_modules": ["q_proj"], "base_model_name_or_path": "org/base"}

    back = peft.peft_import(str(out))
    assert back["lora"]["rank"] == 2 and back["lora"]["scale"] == 4.0
    (tmp_path / "back.safetensors").write_bytes(back["data"])
    ours = dict(mx.load(str(tmp_path / "back.safetensors")))
    assert np.array_equal(np.array(ours["model.layers.0.self_attn.q_proj.lora_a"]), np.array(a))
    assert np.array_equal(np.array(ours["model.layers.0.self_attn.q_proj.lora_b"]), np.array(b))
