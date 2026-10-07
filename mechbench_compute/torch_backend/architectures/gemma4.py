from __future__ import annotations

from functools import partial
from typing import Any

from mechbench_compute._arch import Arch
from mechbench_compute.adapter_keys import ADAPTER_KEYS
from mechbench_compute.support import (
    CORE_GLOBAL_POINTS,
    CORE_LAYER_POINTS,
    Absence,
    Architecture,
    Refusal,
    refuse_head_weights,
)
from mechbench_compute.thinking import Delimiters
from mechbench_compute.tool_dialects.gemma4 import DIALECT
from mechbench_compute.torch_backend.architectures._decoder import (
    read_language_model,
    read_text,
    read_text_path,
)
from mechbench_compute.torch_backend.decoding import make_prompt_cache
from mechbench_compute.torch_backend.forward import Step, make_source_step, run_forward
from mechbench_compute.torch_backend.heads import (
    make_head_logits,
    make_project_to_logits,
    make_unembed,
)
from mechbench_compute.torch_backend.loading import (
    load_transformers,
    read_text_config,
    tokenize_chat,
)
from mechbench_compute.torch_backend.norms import TorchNorm
from mechbench_compute.torch_backend.sites import Sites, read_child_output


def read_arch(model: Any, model_id: str | None = None) -> Arch:
    cfg = read_text_config(model.config)
    layers = cfg.per_layer_config
    layer_types = list(cfg.layer_types)
    n_layers = len(layer_types)
    global_layers = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    sliding = next((i for i, t in enumerate(layer_types) if t != "full_attention"), 0)
    global_kv = (layers[global_layers[0]].num_key_value_heads
                 if getattr(cfg, "attention_k_eq_v", False) and global_layers else None)
    return Arch(
        model_id=model_id or getattr(model.config, "_name_or_path", "") or "",
        n_layers=n_layers,
        d_model=int(cfg.hidden_size),
        n_heads=int(cfg.num_attention_heads),
        n_kv_heads=int(layers[sliding].num_key_value_heads),
        vocab_size=int(cfg.vocab_size),
        hidden_size_per_layer_input=int(getattr(cfg, "hidden_size_per_layer_input", 0) or 0),
        global_layers=global_layers,
        first_kv_shared_layer=n_layers - int(getattr(cfg, "num_kv_shared_layers", 0) or 0),
        model_type="gemma4",
        n_global_kv_heads=None if global_kv is None else int(global_kv),
    )


def read_layer_scalars(model: Any) -> tuple[float, ...]:
    return tuple(float(layer.layer_scalar.float().item()) for layer in read_text(model).layers)


def read_attn_out_norm(model: Any, layer: int) -> TorchNorm:
    return TorchNorm(read_text(model).layers[layer].post_attention_layernorm, 0.0)


def plan_qkv(attn: Any, i: int, points: set[str]) -> list[Step]:
    source = attn.source
    if attn._module.is_kv_shared_layer:
        ops = {"attn.q": source.query_states_transpose_0, "attn.k": source.key_states_to_0,
               "attn.v": source.value_states_to_0}
    else:
        ops = {"attn.q": source.query_states_transpose_0, "attn.k": source.key_states_transpose_0,
               "attn.v": source.value_states_transpose_0}
    return [make_source_step(op, i, point) for point, op in ops.items() if point in points]


SITES = Sites(
    text_path=read_text_path,
    attn_out=partial(read_child_output, "post_attention_layernorm"),
    mlp_out=partial(read_child_output, "post_feedforward_layernorm"),
    gate_out=partial(read_child_output, "post_per_layer_input_norm"),
    qkv=plan_qkv,
)

read_unembed = make_unembed(read_text, 0.0)

ARCH = Architecture(
    model_type="gemma4", name="Gemma 4", loader="transformers",
    generate=True, score=True, train=True,
    layer_points=(*CORE_LAYER_POINTS, "gate_out"), global_points=CORE_GLOBAL_POINTS,
    residual_law=("resid_post[i] == (resid_pre[i] + attn_out[i] + mlp_out[i] + gate_out[i])"
                  " * layer_scalar[i] == resid_pre[i+1]"),
    load=partial(load_transformers, classes={"gemma4": "Gemma4ForConditionalGeneration"}),
    arch_of=read_arch,
    forward=partial(run_forward, sites=SITES),
    lm=read_language_model,
    prompt_cache=make_prompt_cache,
    head_logits=make_head_logits(read_unembed),
    project_to_logits=make_project_to_logits(read_unembed),
    tokenize=tokenize_chat,
    attribution_unembed=read_unembed,
    head_weights=refuse_head_weights("gemma4", "torch"),
    dialect=DIALECT,
    reasoning=(Delimiters("<|channel>", "<channel|>", "thought\n"),),
    adapter_keys=ADAPTER_KEYS,
    refused_when=(Refusal(
        "enable_moe_block",
        "mixture-of-experts layers (Gemma 4 26B A4B) are not wired into the "
        "gemma4 forward; only the dense checkpoints load"),),
    absent_when=(Absence(
        "gate_out", "hidden_size_per_layer_input",
        "this checkpoint has no per-layer input embeddings (hidden_size_per_layer_input "
        "is 0, as in Gemma 4 31B), so no layer adds a per-layer gate"),),
    layer_scalars=read_layer_scalars,
    attn_out_norm=read_attn_out_norm,
    backend="torch",
)
