from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mechbench_compute.architectures import BY_MODEL_TYPE as ON_MLX
from mechbench_compute.model import Model
from mechbench_compute.torch_backend.architectures._decoder import read_text
from tests.tiny_tokenizer import build_tokenizer
from tests.tiny_torch_models import KIT_MODELS


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
