from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelRef:
    base_kind: str
    base: str
    adapter_labels: tuple[str, ...] = ()
    adapter_payloads: tuple[Mapping[str, Any], ...] = field(default=(), compare=False)
    provider: str = ""
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_endpoint(self) -> bool:
        return self.base_kind == "endpoint"

    def to_wire(self) -> dict[str, Any]:
        if self.is_endpoint:
            out: dict[str, Any] = {"provider": self.provider, "model": self.base}
            if self.provider_options:
                out["provider_options"] = dict(self.provider_options)
            return out
        return {
            "base": {self.base_kind: self.base},
            "adapters": [{"bench": label} for label in self.adapter_labels],
        }

    def describe(self) -> str:
        if self.is_endpoint:
            return f"{self.provider}:{self.base}"
        tail = ""
        if self.adapter_labels:
            n = len(self.adapter_labels)
            tail = f" (+{n} adapter{'s' if n != 1 else ''})"
        return f"{self.base_kind}:{self.base}{tail}"


def parse(value: Any) -> ModelRef:
    if isinstance(value, str):
        if not value:
            raise ValueError("model reference is empty")
        return ModelRef(base_kind="hf", base=value)
    if isinstance(value, ModelRef):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"a model reference is a string or an object, not {type(value).__name__}"
        )
    if "provider" in value and "base" not in value:
        from mechbench_compute.providers import PROVIDERS

        provider = str(value["provider"])
        if provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r} in a model reference — "
                f"known: {', '.join(PROVIDERS)}")
        model = value.get("model")
        if not model:
            raise ValueError(
                f'an endpoint reference needs a model: {{"provider": '
                f'"{provider}", "model": "..."}}')
        if value.get("adapters"):
            raise ValueError(
                "an endpoint reference cannot carry adapters — someone "
                "else runs those weights, and a LoRA of ours cannot be "
                "fused into them")
        return ModelRef(base_kind="endpoint", base=str(model), provider=provider,
                        provider_options=dict(value.get("provider_options") or {}))
    base = value.get("base")
    if isinstance(base, str):
        base_kind, base_val = "hf", base
    elif isinstance(base, Mapping) and set(base.keys()) == {"hf"}:
        base_kind, base_val = "hf", str(base["hf"])
    elif isinstance(base, Mapping) and set(base.keys()) == {"bench"}:
        base_kind, base_val = "bench", str(base["bench"])
    else:
        raise ValueError(
            'model reference needs base: "repo[@rev]", {"hf": ...} or {"bench": ...}'
        )
    raw = value.get("adapters", [])
    if not isinstance(raw, (list, tuple)):
        raise TypeError("model reference adapters must be a list")
    labels: list[str] = []
    for a in raw:
        if isinstance(a, Mapping) and set(a.keys()) == {"bench"}:
            labels.append(str(a["bench"]))
        elif isinstance(a, str):
            labels.append(a)
        else:
            raise ValueError(
                'each adapter must be {"bench": "<label>"} (or a bare label)'
            )
    return ModelRef(base_kind=base_kind, base=base_val, adapter_labels=tuple(labels))


def resolve(
    value: Any,
    fetch: Callable[[str], Mapping[str, Any]],
) -> ModelRef:
    ref = parse(value)
    if ref.is_endpoint:
        return ref
    if len(ref.adapter_labels) > 8:
        raise ValueError(
            f"adapter stack of depth {len(ref.adapter_labels)} — the cap "
            "is 8; merge earlier rounds into a checkpoint instead"
        )
    payloads = tuple(fetch(label) for label in ref.adapter_labels)
    for label, p in zip(ref.adapter_labels, payloads):
        if not isinstance(p, Mapping) or "data" not in p:
            raise ValueError(
                f"adapter {label!r} resolved to something without "
                "safetensors bytes under 'data' — is it an adapter object?"
            )
    return ModelRef(
        base_kind=ref.base_kind,
        base=ref.base,
        adapter_labels=ref.adapter_labels,
        adapter_payloads=payloads,
    )
