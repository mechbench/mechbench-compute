from __future__ import annotations

from functools import partial
from typing import Any

from mechbench_compute.adapter_keys import ADAPTER_KEYS
from mechbench_compute.support import (
    CORE_GLOBAL_POINTS,
    CORE_LAYER_POINTS,
    Architecture,
    refuse_head_weights,
    refuse_logit_softcap,
)
from mechbench_compute.torch_backend.forward import run_forward
from mechbench_compute.torch_backend.heads import (
    make_head_logits,
    make_project_to_logits,
    make_unembed,
    refuse_prompt_cache,
)
from mechbench_compute.torch_backend.loading import (
    load_transformers,
    read_hf_arch,
    tokenize_chat,
)
from mechbench_compute.torch_backend.norms import TorchNorm
from mechbench_compute.torch_backend.sites import Sites, read_child_output


def read_text_path(model: Any) -> tuple[str, ...]:
    return ("model", "language_model") if hasattr(model.model, "language_model") else ("model",)


def read_text(model: Any) -> Any:
    found = model
    for name in read_text_path(model):
        found = getattr(found, name)
    return found


def read_language_model(model: Any) -> Any:
    if read_text_path(model) == ("model",):
        return model
    from types import SimpleNamespace

    return SimpleNamespace(model=model.model.language_model, lm_head=model.lm_head)


def read_arch(model: Any, model_id: str | None = None) -> Any:
    return read_hf_arch("gemma3", model.config, model_id)


def read_attn_out_norm(model: Any, layer: int) -> TorchNorm:
    return TorchNorm(read_text(model).layers[layer].post_attention_layernorm, 1.0)


SITES = Sites(
    text_path=read_text_path,
    attn_out=partial(read_child_output, "post_attention_layernorm"),
    mlp_out=partial(read_child_output, "post_feedforward_layernorm"),
)

read_unembed = make_unembed(read_text, 1.0)

ARCH = Architecture(
    model_type="gemma3", name="Gemma 3", loader="transformers",
    generate=False, score=False, train=False,
    layer_points=CORE_LAYER_POINTS, global_points=CORE_GLOBAL_POINTS,
    residual_law="resid_post[i] == resid_pre[i] + attn_out[i] + mlp_out[i] == resid_pre[i+1]",
    load=partial(load_transformers, classes={"gemma3": "Gemma3ForConditionalGeneration"}),
    arch_of=read_arch,
    forward=partial(run_forward, sites=SITES),
    lm=read_language_model,
    prompt_cache=refuse_prompt_cache,
    head_logits=make_head_logits(read_unembed),
    project_to_logits=make_project_to_logits(read_unembed),
    tokenize=tokenize_chat,
    attribution_unembed=read_unembed,
    head_weights=refuse_head_weights("gemma3"),
    dialect=None,
    reasoning=(),
    adapter_keys=ADAPTER_KEYS,
    refused_when=(refuse_logit_softcap("Gemma 3"),),
    attn_out_norm=read_attn_out_norm,
    backend="torch",
)
