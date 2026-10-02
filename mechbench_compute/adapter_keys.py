from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterKeys:
    key_re: re.Pattern
    containers: Mapping[str, str]
    peft_re: re.Pattern


ADAPTER_KEYS = AdapterKeys(
    key_re=re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([ab])$"),
    containers={
        "q_proj": "self_attn", "k_proj": "self_attn",
        "v_proj": "self_attn", "o_proj": "self_attn",
        "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp",
    },
    peft_re=re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$"),
)


def check_layers(layers: Sequence[int], n_layers: int) -> None:
    beyond = [i for i in layers if not 0 <= i < n_layers]
    if beyond:
        raise ValueError(
            f"LAYER_OUT_OF_RANGE: an adapter's `layers` names "
            f"{', '.join(str(i) for i in beyond)}, and this model's layers are "
            f"0 through {n_layers - 1}")


def group_by_module(weights: Mapping[str, Any], keys: AdapterKeys = ADAPTER_KEYS,
                    layers: Sequence[int] | None = None) -> dict[tuple[int, str, str], dict[str, Any]]:
    pairs: dict[tuple[int, str, str], dict[str, Any]] = {}
    for key, w in weights.items():
        m = keys.key_re.match(key)
        if m is None:
            raise ValueError(f"unrecognized adapter key {key!r}")
        i, container, proj, ab = (int(m.group(1)), m.group(2),
                                  m.group(3), m.group(4))
        if layers is None or i in layers:
            pairs.setdefault((i, container, proj), {})[ab] = w
    return pairs
