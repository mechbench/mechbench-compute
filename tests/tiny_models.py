from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_map

from mechbench_compute.model import Model

WINDOW = 3

WORDS = ("<pad>", "<unk>", "<start>", "<end>", "user", "model", "assistant",
         "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran")

CHAT_TEMPLATE = (
    "{% for m in messages %}<start>{{ m['role'] }} "
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for c in m['content'] %}{% if c['type'] == 'text' %}{{ c['text'] }}"
    "{% endif %}{% endfor %}{% endif %}<end>{% endfor %}"
    "{% if add_generation_prompt %}<start>model {% endif %}")


def build_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
        bos_token="<start>", eos_token="<end>",
        additional_special_tokens=["<start>", "<end>"], chat_template=CHAT_TEMPLATE)


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
