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
from mechbench_compute.thinking import THINK_TAGS
from mechbench_compute.tool_dialects.llama import DIALECT
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
from mechbench_compute.torch_backend.sites import (
    Sites,
    read_child_first_output,
    read_child_output,
)


def read_text_path(model: Any) -> tuple[str, ...]:
    return ("model",)


def read_text(model: Any) -> Any:
    return model.model


def read_language_model(model: Any) -> Any:
    return model


def read_arch(model: Any, model_id: str | None = None) -> Any:
    return read_hf_arch("llama", model.config, model_id)


SITES = Sites(
    text_path=read_text_path,
    attn_out=partial(read_child_first_output, "self_attn"),
    mlp_out=partial(read_child_output, "mlp"),
)

read_unembed = make_unembed(read_text, 0.0)

ARCH = Architecture(
    model_type="llama", name="Llama", loader="transformers",
    generate=False, score=False, train=False,
    layer_points=CORE_LAYER_POINTS, global_points=CORE_GLOBAL_POINTS,
    residual_law="resid_post[i] == resid_pre[i] + attn_out[i] + mlp_out[i] == resid_pre[i+1]",
    load=partial(load_transformers, classes={}),
    arch_of=read_arch,
    forward=partial(run_forward, sites=SITES),
    lm=read_language_model,
    prompt_cache=refuse_prompt_cache,
    head_logits=make_head_logits(read_unembed),
    project_to_logits=make_project_to_logits(read_unembed),
    tokenize=partial(tokenize_chat, special=True),
    attribution_unembed=read_unembed,
    head_weights=refuse_head_weights("llama"),
    dialect=DIALECT,
    reasoning=(THINK_TAGS,),
    adapter_keys=ADAPTER_KEYS,
    refused_when=(refuse_logit_softcap("Llama"),),
    backend="torch",
)
