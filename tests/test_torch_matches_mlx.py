from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None or importlib.util.find_spec("mlx") is None:
    pytest.skip("the cross-backend check needs both torch (with nnsight) and MLX",
                allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from safetensors.torch import save as save_tensors

from mechbench_compute.architectures import BY_MODEL_TYPE as ON_MLX
from mechbench_compute.arrays import read_f64
from mechbench_compute.lora import fuse_adapter_stack as fuse_on_mlx
from mechbench_compute.model import Model
from mechbench_compute.torch_backend.architectures._decoder import read_text
from mechbench_compute.torch_backend.lora import fuse_adapter_stack as fuse_on_torch
from tests.tiny_tokenizer import build_tokenizer
from tests.tiny_torch_models import KIT_MODELS, build_tiny_model, make_adapter

IDS = [1, 5, 9, 2, 7, 3, 11, 4]

TOLERANCE = 1e-5

# external: mlx-vlm models/gemma3/language.py — rounds Gemma 3's embedding scale to bf16 where transformers keeps the weights' dtype, so float32 weights part by about 1e-4 from the embedding on
TOLERANCE_OF = {"gemma3": 1e-3}


def build_gemma4(cfg):
    from mlx_vlm.models.gemma4 import config

    from tests.tiny_models import wrap_gemma4

    first_global = cfg.layer_types.index("full_attention")
    k_eq_v = bool(cfg.attention_k_eq_v)
    return wrap_gemma4(config.TextConfig(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        intermediate_size=cfg.intermediate_size, num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.per_layer_config[0].num_key_value_heads,
        num_global_key_value_heads=(cfg.per_layer_config[first_global].num_key_value_heads
                                    if k_eq_v else None),
        head_dim=cfg.per_layer_config[0].head_dim,
        global_head_dim=cfg.per_layer_config[first_global].head_dim,
        vocab_size=cfg.vocab_size, vocab_size_per_layer_input=cfg.vocab_size_per_layer_input,
        hidden_size_per_layer_input=cfg.hidden_size_per_layer_input,
        num_kv_shared_layers=cfg.num_kv_shared_layers, use_double_wide_mlp=cfg.use_double_wide_mlp,
        attention_k_eq_v=k_eq_v, sliding_window=cfg.sliding_window,
        layer_types=list(cfg.layer_types), rope_parameters=dict(cfg.rope_parameters),
        final_logit_softcapping=cfg.final_logit_softcapping, rms_norm_eps=cfg.rms_norm_eps))


def build_gemma3(cfg):
    from types import SimpleNamespace

    from mlx_vlm.models.gemma3 import config, language

    text = config.TextConfig(
        model_type="gemma3_text", hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers, intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, vocab_size=cfg.vocab_size, sliding_window=cfg.sliding_window,
        sliding_window_pattern=cfg.layer_types.index("full_attention") + 1,
        query_pre_attn_scalar=cfg.query_pre_attn_scalar, rms_norm_eps=cfg.rms_norm_eps,
        rope_global_base_freq=cfg.rope_parameters["full_attention"]["rope_theta"],
        rope_local_base_freq=cfg.rope_parameters["sliding_attention"]["rope_theta"])
    lm = language.LanguageModel(text)
    return SimpleNamespace(language_model=lm,
                           config=SimpleNamespace(text_config=text, model_type="gemma3")), lm


def build_mlx_lm(cfg):
    from mlx_lm.models import llama, qwen2

    module = {"qwen2": qwen2, "llama": llama}[cfg.model_type]
    model = module.Model(module.ModelArgs(
        model_type=cfg.model_type, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers, intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps, vocab_size=cfg.vocab_size,
        rope_theta=cfg.rope_parameters["rope_theta"],
        tie_word_embeddings=cfg.tie_word_embeddings))
    return model, model


BUILDERS = {"gemma4": build_gemma4, "gemma3": build_gemma3, "qwen2": build_mlx_lm,
            "llama": build_mlx_lm}

PAIRS = [(name, t) for name, t in KIT_MODELS if t in BUILDERS and t in ON_MLX]


def read_torch_weights(model) -> dict:
    lm = model.lm
    out = {f"model.{k}": v for k, v in read_text(model._model).state_dict().items()}
    out["lm_head.weight"] = lm.lm_head.weight
    return {k: v.detach().float().cpu().numpy() for k, v in out.items()}


def build_on_mlx(on_torch) -> Model:
    config = on_torch._model.config
    cfg = getattr(config, "text_config", None) or config
    wrapped, lm = BUILDERS[on_torch.architecture.model_type](cfg)
    weights = read_torch_weights(on_torch)
    wanted = dict(tree_flatten(lm.parameters()))
    missing = sorted(set(wanted) - set(weights))
    assert not missing, missing
    lm.update(tree_unflatten([(k, mx.array(weights[k]).reshape(v.shape)) for k, v in wanted.items()]))
    mx.eval(lm.parameters())
    return Model(wrapped, build_tokenizer(), architecture=ON_MLX[on_torch.architecture.model_type])


def read_shared_names(on_torch, on_mlx) -> list[str]:
    a, b = on_torch.architecture, on_mlx.architecture
    n = on_torch.arch.n_layers
    return ([f"blocks.{i}.{p}" for i in range(n) for p in a.layer_points_of(on_torch.arch)
             if p in b.layer_points_of(on_mlx.arch)]
            + [p for p in a.global_points if p in b.global_points])


@pytest.mark.parametrize("name", [n for n, _ in PAIRS])
def test_the_same_weights_give_the_same_logits_and_points_on_mlx_and_torch(name):
    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    assert on_mlx.arch.n_layers == on_torch.arch.n_layers
    names = read_shared_names(on_torch, on_mlx)
    torch_run = on_torch.run(on_torch.make_ids(IDS), capture=names)
    mlx_run = on_mlx.run(on_mlx.make_ids(IDS), capture=names)
    tolerance = TOLERANCE_OF.get(on_torch.architecture.model_type, TOLERANCE)
    want, got = read_f64(mlx_run.logits), read_f64(torch_run.logits)
    assert want.shape == got.shape and np.abs(want).max() > 0.25
    assert np.abs(got - want).max() <= tolerance * max(1.0, np.abs(want).max()), (
        f"{name}: logits differ by {np.abs(got - want).max():.3g}")
    for point in names:
        a, b = read_f64(mlx_run.cache[point]), read_f64(torch_run.cache[point])
        assert a.shape == b.shape, (point, a.shape, b.shape)
        err = float(np.abs(a - b).max())
        assert err <= tolerance * max(1.0, float(np.abs(a).max())), f"{name} {point}: {err:.3g}"


@pytest.mark.parametrize("name", [n for n, _ in PAIRS])
def test_the_same_stored_adapter_moves_the_same_weights_on_mlx_and_torch(name):
    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    weights = make_adapter(on_torch, on_torch.architecture.adapter_keys)
    payload = {"data": save_tensors(weights), "lora": {"rank": 2, "alpha": 4.0}}
    torch_before = read_torch_weights(on_torch)
    mlx_before = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.parameters())}
    fuse_on_torch(on_torch.lm, [payload], layers=[[0, 2]])
    fuse_on_mlx(on_mlx.lm, [payload], layers=[[0, 2]])
    mx.eval(on_mlx.lm.parameters())
    mlx_after = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.parameters())}
    torch_after = read_torch_weights(on_torch)
    stems = [k.removesuffix(".lora_a") for k in weights if k.endswith(".lora_a")]
    assert len(stems) >= 3 * on_torch.arch.n_layers
    for stem in stems:
        name = f"{stem}.weight"
        moved_torch = torch_after[name] - torch_before[name]
        moved_mlx = mlx_after[name] - mlx_before[name]
        assert np.allclose(moved_torch, moved_mlx, atol=1e-5), name
        assert bool(moved_mlx.any()) == (int(stem.split(".")[2]) in (0, 2)), name
