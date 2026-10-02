from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute._arch import Arch


def read_text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


def read_hf_arch(model_type: str, config: Any, model_id: str | None = None) -> Arch:
    cfg = read_text_config(config)
    n_layers = int(cfg.num_hidden_layers)
    layer_types = getattr(cfg, "layer_types", None)
    global_layers = (tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
                     if layer_types else tuple(range(n_layers)))
    n_heads = int(cfg.num_attention_heads)
    return Arch(
        model_id=model_id or getattr(config, "_name_or_path", "") or "",
        n_layers=n_layers,
        d_model=int(cfg.hidden_size),
        n_heads=n_heads,
        n_kv_heads=int(getattr(cfg, "num_key_value_heads", None) or n_heads),
        vocab_size=int(cfg.vocab_size),
        hidden_size_per_layer_input=0,
        global_layers=global_layers,
        first_kv_shared_layer=n_layers,
        model_type=model_type,
    )


def read_device(model: Any) -> Any:
    return next(model.parameters()).device


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_transformers(model_id: str, *, classes: Mapping[str, str], device: str | None = None,
                      dtype: Any = None) -> tuple[Any, Any]:
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_id)
    name = classes.get(config.model_type, "AutoModelForCausalLM")
    model = getattr(transformers, name).from_pretrained(
        model_id, dtype=dtype if dtype is not None else torch.bfloat16)
    model.eval().to(device or pick_device())
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def tokenize_chat(model: Any, processor: Any, prompt: str, *, chat_template: bool = True,
                  special: bool | None = None) -> Any:
    import torch

    tok = getattr(processor, "tokenizer", processor)
    if chat_template:
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
        add = special if special is not None else getattr(tok, "chat_template", None) is None
        ids = tok.encode(rendered, add_special_tokens=add)
    else:
        ids = tok.encode(prompt)
    return torch.tensor([ids], dtype=torch.long, device=read_device(model))
