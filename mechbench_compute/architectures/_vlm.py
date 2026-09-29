from __future__ import annotations

from typing import Any

import mlx.core as mx

from mechbench_compute._arch import Arch


def load_vlm(model_id: str, **config: Any) -> tuple[Any, Any]:
    from mlx_vlm import load

    return load(model_id, **config)


def read_language_model(model):
    return model.language_model


def make_vlm_cache(model):
    return model.language_model.make_cache()


def tokenize_vlm(model, processor, prompt: str, *, chat_template: bool = True) -> mx.array:
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    if not chat_template:
        tok = getattr(processor, "tokenizer", processor)
        ids = tok.encode(prompt)
        return mx.array([ids], dtype=mx.int32)

    add_special_tokens = getattr(processor, "chat_template", None) is None
    formatted = apply_chat_template(processor, model.config, prompt, num_images=0)
    image_token_index = getattr(model.config, "image_token_index", None)
    inputs = prepare_inputs(
        processor,
        images=None,
        audio=None,
        prompts=formatted,
        image_token_index=image_token_index,
        resize_shape=None,
        add_special_tokens=add_special_tokens,
    )
    return inputs["input_ids"]


def read_vlm_arch(model_type: str, cfg: Any, model_id: str | None = None) -> Arch:
    if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
        layer_types = list(cfg.layer_types)
        n_layers = len(layer_types)
        global_layers = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    else:
        n_layers = int(cfg.num_hidden_layers)
        pattern = int(getattr(cfg, "sliding_window_pattern", 6))
        global_layers = tuple(i for i in range(n_layers) if (i + 1) % pattern == 0)
    num_kv_shared = int(getattr(cfg, "num_kv_shared_layers", 0) or 0)
    return Arch(
        model_id=model_id or getattr(cfg, "_name_or_path", "") or "",
        n_layers=n_layers,
        d_model=int(cfg.hidden_size),
        n_heads=int(cfg.num_attention_heads),
        n_kv_heads=int(cfg.num_key_value_heads),
        vocab_size=int(cfg.vocab_size),
        hidden_size_per_layer_input=int(getattr(cfg, "hidden_size_per_layer_input", 0) or 0),
        global_layers=global_layers,
        first_kv_shared_layer=n_layers - num_kv_shared,
        model_type=model_type,
    )

