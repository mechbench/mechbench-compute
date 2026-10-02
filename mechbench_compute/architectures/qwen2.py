from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import partial

from mlx_lm.models.base import create_attention_mask

from mechbench_compute._arch import Arch
from mechbench_compute.architectures._head import (
    make_head_logits,
    make_project_to_logits,
    refuse_head_weights,
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
from mechbench_compute.dialects import (
    ParseResult,
    ToolDialect,
    make_call,
    parse_json,
    strip_calls,
)
from mechbench_compute.lora import ADAPTER_KEYS
from mechbench_compute.support import (
    CORE_GLOBAL_POINTS,
    CORE_LAYER_POINTS,
    Architecture,
    refuse_logit_softcap,
)
from mechbench_compute.thinking import THINK_TAGS
from mechbench_compute.tools import ToolDef

CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_calls(text: str, tools: Sequence[ToolDef]) -> ParseResult:
    known = {t.name for t in tools}
    out = []
    spans: list[tuple[int, int]] = []
    for m in CALL.finditer(text):
        parsed = parse_json(m.group(1))
        if (isinstance(parsed, Mapping) and parsed.get("name")
                and not (known and parsed["name"] not in known)):
            args = parsed.get("arguments")
            out.append(make_call(str(parsed["name"]),
                                 dict(args) if isinstance(args, Mapping) else {},
                                 len(out)))
            spans.append(m.span())
    return strip_calls(text, spans), out


def make_masks(tm, h, kv_cache) -> list:
    mask = create_attention_mask(h, kv_cache[0])
    return [mask] * len(tm.layers)


def read_arch(model, model_id: str | None = None) -> Arch:
    return read_lm_arch("qwen2", model, model_id)


ARCH = Architecture(
    model_type="qwen2", name="Qwen 2", loader="mlx-lm",
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
    head_weights=refuse_head_weights("qwen2"),
    dialect=ToolDialect("qwen-2.5", "<tool_call>", parse_calls,
                        attempting=("<tool_call>", '"name"')),
    reasoning=(THINK_TAGS,),
    adapter_keys=ADAPTER_KEYS,
    refused_when=(refuse_logit_softcap("Qwen 2"),),
)
