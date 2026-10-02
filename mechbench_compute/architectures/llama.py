from __future__ import annotations

from functools import partial

from mlx_lm.models.base import create_attention_mask

from mechbench_compute._arch import Arch
from mechbench_compute.adapter_keys import ADAPTER_KEYS
from mechbench_compute.architectures._head import (
    make_head_logits,
    make_project_to_logits,
)
from mechbench_compute.architectures._mlx_lm import (
    load_lm,
    make_lm_cache,
    read_lm,
    read_lm_arch,
    read_unembed,
    run_lm_forward,
    tokenize_lm,
)
from mechbench_compute.support import (
    CORE_GLOBAL_POINTS,
    CORE_LAYER_POINTS,
    Architecture,
    refuse_head_weights,
    refuse_logit_softcap,
)
from mechbench_compute.thinking import THINK_TAGS
from mechbench_compute.tool_dialects.llama import DIALECT


def make_masks(tm, h, kv_cache) -> list:
    fa_mask = create_attention_mask(h, kv_cache[tm.fa_idx])
    swa_mask = None
    if tm.swa_idx is not None:
        swa_mask = create_attention_mask(h, kv_cache[tm.swa_idx], window_size=tm.sliding_window)
    return [swa_mask if layer.use_sliding else fa_mask for layer in tm.layers]


def read_arch(model, model_id: str | None = None) -> Arch:
    return read_lm_arch("llama", model, model_id)


ARCH = Architecture(
    model_type="llama", name="Llama", loader="mlx-lm",
    generate=True, score=True, train=True,
    layer_points=CORE_LAYER_POINTS, global_points=CORE_GLOBAL_POINTS,
    residual_law="resid_post[i] == resid_pre[i] + attn_out[i] + mlp_out[i] == resid_pre[i+1]",
    load=load_lm,
    arch_of=read_arch,
    forward=partial(run_lm_forward, make_masks=make_masks),
    lm=read_lm,
    prompt_cache=make_lm_cache,
    head_logits=make_head_logits(read_unembed),
    project_to_logits=make_project_to_logits(read_unembed),
    tokenize=tokenize_lm,
    attribution_unembed=read_unembed,
    head_weights=refuse_head_weights("llama"),
    dialect=DIALECT,
    reasoning=(THINK_TAGS,),
    adapter_keys=ADAPTER_KEYS,
    refused_when=(refuse_logit_softcap("Llama"),),
)
