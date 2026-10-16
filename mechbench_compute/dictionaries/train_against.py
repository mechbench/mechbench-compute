from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mechbench_compute import model_ref


@dataclass(frozen=True)
class Against:
    swap: Callable[[], contextlib.AbstractContextManager]
    adapters: list[dict[str, Any]]
    model_adapters: list[dict[str, Any]]
    wire: Any


def read_ref(value: Any) -> model_ref.ModelRef | None:
    return None if value is None else model_ref.parse(value)


def read_against(model: Any, given_model: Any, given_against: Any,
                 fused: Sequence[tuple[Any, Sequence[Any], Any]]) -> Against:
    ref = read_ref(given_model)
    labels = tuple(ref.adapter_labels) if ref is not None else ()
    layers = tuple(ref.adapter_layers) if ref is not None else ()
    if given_against is None:
        keep = 0
        other = (model_ref.ModelRef(base_kind=ref.base_kind, base=ref.base) if ref is not None else None)
    else:
        other = read_ref(given_against)
        if ref is None or other.is_endpoint or (other.base_kind, other.base) != (ref.base_kind, ref.base):
            raise ValueError(
                f"dictionary/train: `against` is {other.describe()}, and a crosscoder runs both models on "
                f"the one loaded checkpoint: `against` is "
                f"{ref.describe() if ref is not None else '`model`'} with fewer of its adapters")
        keep = len(other.adapter_labels)
        if (tuple(other.adapter_labels), tuple(other.adapter_layers)) != (labels[:keep], layers[:keep]):
            raise ValueError(
                f"dictionary/train: `against` is {other.describe()}; it is `model` "
                f"({ref.describe()}) with the adapters after its first {keep} taken off, so its adapters "
                f"are the first of `model`'s, in order, each with the same `layers`")
    if keep >= len(labels):
        named = ref.describe() if ref is not None else "the loaded model"
        raise ValueError(
            f"dictionary/train: a crosscoder reads two models, and `model` ({named}) and `against` are "
            "the same one. Give `model` the organism, a reference carrying its adapters; `against` is "
            "that reference without them unless named")
    handles = [h for m, stack, _ in fused if m is model for h in stack]
    if len(handles) != len(labels):
        raise ValueError(
            f"dictionary/train: `model` carries {len(labels)} adapters, and this node fused "
            f"{len(handles)} of them, so it cannot take them off to run `against`. Give the adapted model "
            "as the node's `model` parameter, which the executor resolves and fuses")
    top = handles[keep:]
    return Against(swap=lambda: unfused(model, top), adapters=other.read_adapters(),
                   model_adapters=ref.read_adapters(), wire=other.to_wire())


def unfused(model: Any, handles: Sequence[Any]) -> contextlib.AbstractContextManager:
    from mechbench_compute.lora import unfused_stack

    return unfused_stack(model.lm, handles)
