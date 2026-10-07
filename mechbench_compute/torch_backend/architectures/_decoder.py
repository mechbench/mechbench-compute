from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any

from mechbench_compute.torch_backend.heads import make_unembed
from mechbench_compute.torch_backend.sites import (
    Sites,
    read_child_first_output,
    read_child_output,
)


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
    return SimpleNamespace(model=model.model.language_model, lm_head=model.lm_head)


SITES = Sites(
    text_path=read_text_path,
    attn_out=partial(read_child_first_output, "self_attn"),
    mlp_out=partial(read_child_output, "mlp"),
)

read_unembed = make_unembed(read_text, 0.0)
