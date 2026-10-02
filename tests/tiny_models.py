from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_map

from mechbench_compute.model import Model
from tests.kit_backends import WINDOW
from tests.tiny_tokenizer import build_tokenizer


def build_mlx_lm(model_type):
    from mlx_lm.models import llama, qwen2

    module = {"qwen2": qwen2, "llama": llama}[model_type]
    kw = dict(model_type=model_type, hidden_size=32, num_hidden_layers=4,
              intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
              rms_norm_eps=1e-6, vocab_size=64, tie_word_embeddings=False)
    if model_type == "llama":
        kw.update(layer_types=["sliding_attention", "full_attention"] * 2,
                  sliding_window=WINDOW)
    model = module.Model(module.ModelArgs(**kw))
    return model, model


def build_gemma3():
    from mlx_vlm.models.gemma3 import config, language

    cfg = config.TextConfig(model_type="gemma3_text", hidden_size=32, num_hidden_layers=4,
                            intermediate_size=64, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, vocab_size=64,
                            sliding_window=WINDOW, sliding_window_pattern=2,
                            query_pre_attn_scalar=8)
    lm = language.LanguageModel(cfg)
    wrapped = SimpleNamespace(language_model=lm,
                              config=SimpleNamespace(text_config=cfg, model_type="gemma3"))
    return wrapped, lm


def build_gemma4():
    from mlx_vlm.models.gemma4 import config

    return wrap_gemma4(config.TextConfig(
        hidden_size=32, num_hidden_layers=4, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=8, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, num_kv_shared_layers=0,
        sliding_window=WINDOW, sliding_window_pattern=2))


def build_gemma4_31b():
    from mlx_vlm.models.gemma4 import config

    return wrap_gemma4(config.TextConfig(
        hidden_size=32, num_hidden_layers=6, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
        head_dim=8, global_head_dim=16, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=0, num_kv_shared_layers=0, attention_k_eq_v=True,
        layer_types=["sliding_attention"] * 5 + ["full_attention"], sliding_window=WINDOW,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                               "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}))


def wrap_gemma4(text):
    from mlx_vlm.models.gemma4 import config, gemma4

    vision = config.VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                                 global_head_dim=8, position_embedding_size=16)
    model = gemma4.Model(config.ModelConfig(text_config=text, vision_config=vision,
                                            vocab_size=64, image_token_id=62,
                                            audio_token_id=63))
    return model, model.language_model


BUILDERS = {
    "gemma3": build_gemma3,
    "gemma4": build_gemma4,
    "llama": lambda: build_mlx_lm("llama"),
    "qwen2": lambda: build_mlx_lm("qwen2"),
}

MODEL_TYPES = tuple(sorted(BUILDERS))

VARIANTS = {"gemma4-31b": ("gemma4", build_gemma4_31b)}

KIT_MODELS = (*((t, t) for t in MODEL_TYPES), *((v, t) for v, (t, _) in VARIANTS.items()))


def build_tiny_model(name, architecture=None) -> Model:
    build = VARIANTS[name][1] if name in VARIANTS else BUILDERS[name]
    wrapped, lm = build()
    keys = iter(mx.random.split(mx.random.key(11), 1000))
    lm.update(tree_map(lambda p: mx.random.normal(p.shape, key=next(keys)) * 0.5,
                       lm.parameters()))
    mx.eval(lm.parameters())
    return Model(wrapped, build_tokenizer(), architecture=architecture)


def read_parameter_names(model) -> set[str]:
    from mlx.utils import tree_flatten

    return set(dict(tree_flatten(model.lm.parameters())))


def fit_adapter(model, keys) -> int:
    from mechbench_compute.lora import apply_lora

    return apply_lora(model.lm, rank=2, targets=tuple(keys.containers), keys=keys)


def read_config(model) -> dict:
    config = getattr(model._model, "config", None)
    source = getattr(config, "text_config", None) or getattr(model._model, "args", None)
    return dict(vars(source)) if source is not None else {}


def make_kit():
    from mechbench_compute import backends
    from mechbench_compute.architectures import BY_MODEL_TYPE
    from tests.kit_backends import KitBackend

    return KitBackend(
        backend=next(b for b in backends.BACKENDS if b.name == "mlx"),
        capabilities=backends.advertise(),
        architectures=BY_MODEL_TYPE,
        models=KIT_MODELS,
        build=build_tiny_model,
        read_parameter_names=read_parameter_names,
        fit_adapter=fit_adapter,
        read_config=read_config)


KIT = make_kit()
