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
from mechbench_compute.tool_dialects.qwen2 import DIALECT
from mechbench_compute.torch_backend.architectures._decoder import (
    SITES,
    read_language_model,
    read_unembed,
)
from mechbench_compute.torch_backend.decoding import make_prompt_cache
from mechbench_compute.torch_backend.forward import run_forward
from mechbench_compute.torch_backend.heads import (
    make_head_logits,
    make_project_to_logits,
)
from mechbench_compute.torch_backend.loading import (
    load_transformers,
    read_hf_arch,
    tokenize_chat,
)


def read_arch(model: Any, model_id: str | None = None) -> Any:
    return read_hf_arch("qwen2", model.config, model_id)


ARCH = Architecture(
    model_type="qwen2", name="Qwen 2", loader="transformers",
    generate=True, score=True, train=True,
    layer_points=CORE_LAYER_POINTS, global_points=CORE_GLOBAL_POINTS,
    residual_law="resid_post[i] == resid_pre[i] + attn_out[i] + mlp_out[i] == resid_pre[i+1]",
    load=partial(load_transformers, classes={}),
    arch_of=read_arch,
    forward=partial(run_forward, sites=SITES),
    lm=read_language_model,
    prompt_cache=make_prompt_cache,
    head_logits=make_head_logits(read_unembed),
    project_to_logits=make_project_to_logits(read_unembed),
    tokenize=tokenize_chat,
    attribution_unembed=read_unembed,
    head_weights=refuse_head_weights("qwen2", "torch"),
    dialect=DIALECT,
    reasoning=(THINK_TAGS,),
    adapter_keys=ADAPTER_KEYS,
    refused_when=(refuse_logit_softcap("Qwen 2"),),
    backend="torch",
)
