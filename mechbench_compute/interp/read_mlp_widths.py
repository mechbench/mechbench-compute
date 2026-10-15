from __future__ import annotations


def read_mlp_widths(model, layers: list[int]) -> list[int]:
    blocks = model.lm.model.layers
    return [read_out_width(blocks[i].mlp.gate_proj) for i in layers]


def read_out_width(linear) -> int:
    while hasattr(linear, "base"):
        linear = linear.base
    return int(linear.weight.shape[0])
