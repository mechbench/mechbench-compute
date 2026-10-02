from __future__ import annotations

import os

import torch
from transformers import (
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    SiglipVisionConfig,
)

from mechbench_compute import backends
from mechbench_compute.torch_backend.architectures import BY_MODEL_TYPE
from mechbench_compute.torch_backend.lora import fuse, restore
from mechbench_compute.torch_backend.model import TorchModel
from tests.kit_backends import WINDOW, KitBackend
from tests.tiny_tokenizer import build_tokenizer

DEVICE = os.environ.get("MECHBENCH_KIT_DEVICE", "cpu")

SEED = 11


def make_gemma3_text() -> Gemma3TextConfig:
    return Gemma3TextConfig(hidden_size=32, num_hidden_layers=4, intermediate_size=64,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            vocab_size=64, sliding_window=WINDOW, sliding_window_pattern=2,
                            query_pre_attn_scalar=8, max_position_embeddings=64)


def build_gemma3() -> torch.nn.Module:
    return Gemma3ForCausalLM(make_gemma3_text())


def build_gemma3_vlm() -> torch.nn.Module:
    vision = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, image_size=28, patch_size=14)
    return Gemma3ForConditionalGeneration(Gemma3Config(
        text_config=make_gemma3_text(), vision_config=vision, mm_tokens_per_image=4,
        image_token_index=62, boi_token_index=60, eoi_token_index=61))


def build_llama() -> torch.nn.Module:
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=32, num_hidden_layers=4, intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=64, tie_word_embeddings=False,
        max_position_embeddings=64))


BUILDERS = {"gemma3": build_gemma3, "llama": build_llama}

MODEL_TYPES = tuple(sorted(BUILDERS))

VARIANTS = {"gemma3-vlm": ("gemma3", build_gemma3_vlm)}

KIT_MODELS = (*((t, t) for t in MODEL_TYPES), *((v, t) for v, (t, _) in VARIANTS.items()))


def build_tiny_model(name: str, architecture=None) -> TorchModel:
    build = VARIANTS[name][1] if name in VARIANTS else BUILDERS[name]
    torch.manual_seed(SEED)
    model = build()
    draw = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=draw) * 0.5)
    model.eval().to(DEVICE)
    return TorchModel(model, build_tokenizer(), architecture=architecture)


def read_parameter_names(model: TorchModel) -> set[str]:
    return {f"model.{n}" for n, _ in model.lm.model.named_parameters()}


def make_adapter(model: TorchModel, keys, rank: int = 2) -> dict[str, torch.Tensor]:
    draw = torch.Generator().manual_seed(SEED)
    out: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(model.lm.model.layers):
        for proj, container in keys.containers.items():
            mod = getattr(getattr(layer, container), proj, None)
            if mod is None:
                continue
            d_out, d_in = mod.weight.shape
            stem = f"model.layers.{i}.{container}.{proj}"
            out[f"{stem}.lora_a"] = torch.randn((rank, d_in), generator=draw)
            out[f"{stem}.lora_b"] = torch.randn((d_out, rank), generator=draw)
    return out


def fit_adapter(model: TorchModel, keys) -> int:
    handle = fuse(model.lm, make_adapter(model, keys), 2.0, keys=keys)
    restore(model.lm, handle)
    return len(handle)


def read_config(model: TorchModel) -> dict:
    config = model._model.config
    return (getattr(config, "text_config", None) or config).to_dict()


KIT = KitBackend(
    backend=next(b for b in backends.BACKENDS if b.name == "torch"),
    capabilities={"accelerator": "cuda" if DEVICE.startswith("cuda") else "cpu",
                  "backends": ["torch"]},
    architectures=BY_MODEL_TYPE,
    models=KIT_MODELS,
    build=build_tiny_model,
    read_parameter_names=read_parameter_names,
    fit_adapter=fit_adapter,
    read_config=read_config,
)
