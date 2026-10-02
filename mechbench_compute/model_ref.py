from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any


@dataclass(frozen=True)
class ModelRef:
    base_kind: str
    base: str
    adapter_labels: tuple[str, ...] = ()
    adapter_payloads: tuple[Mapping[str, Any], ...] = field(default=(), compare=False)
    provider: str = ""
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    adapter_layers: tuple[tuple[int, ...] | None, ...] = ()

    def __post_init__(self) -> None:
        short = len(self.adapter_labels) - len(self.adapter_layers)
        if short < 0:
            raise ValueError(
                f"a model reference with {len(self.adapter_labels)} adapters was given "
                f"{len(self.adapter_layers)} sets of layers; one per adapter at most")
        if short:
            object.__setattr__(self, "adapter_layers", (*self.adapter_layers, *(None,) * short))

    @property
    def is_endpoint(self) -> bool:
        return self.base_kind == "endpoint"

    def read_adapters(self) -> list[dict[str, Any]]:
        return [{"bench": label} if layers is None else {"bench": label, "layers": list(layers)}
                for label, layers in zip(self.adapter_labels, self.adapter_layers)]

    def to_wire(self) -> dict[str, Any]:
        if self.is_endpoint:
            out: dict[str, Any] = {"provider": self.provider, "model": self.base}
            if self.provider_options:
                out["provider_options"] = dict(self.provider_options)
            return out
        return {
            "base": {self.base_kind: self.base},
            "adapters": self.read_adapters(),
        }

    def describe(self) -> str:
        if self.is_endpoint:
            return f"{self.provider}:{self.base}"
        tail = ""
        if self.adapter_labels:
            n = len(self.adapter_labels)
            chosen = "".join(f"; {label} in {describe_layers(layers)}"
                             for label, layers in zip(self.adapter_labels, self.adapter_layers)
                             if layers is not None)
            tail = f" (+{n} adapter{'s' if n != 1 else ''}{chosen})"
        return f"{self.base_kind}:{self.base}{tail}"


def describe_layers(layers: tuple[int, ...]) -> str:
    if not layers:
        return "no layer"
    runs: list[list[int]] = []
    for i in layers:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    said = ", ".join(f"{run[0]}–{run[-1]}" if len(run) > 2 else ", ".join(map(str, run))
                     for run in runs)
    return f"layer{'s' if len(layers) > 1 else ''} {said}"


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
    layers: list[tuple[int, ...] | None] = []
    for a in raw:
        if isinstance(a, Mapping) and "bench" in a and set(a.keys()) <= {"bench", "layers"}:
            labels.append(str(a["bench"]))
            layers.append(read_layers(a["layers"], labels[-1]) if "layers" in a else None)
        elif isinstance(a, str):
            labels.append(a)
            layers.append(None)
        else:
            raise ValueError(
                'each adapter must be {"bench": "<label>"} (or a bare label), with '
                '`layers` beside `bench` to fuse it in those layers only'
            )
    return ModelRef(base_kind=base_kind, base=base_val, adapter_labels=tuple(labels),
                    adapter_layers=tuple(layers))


def read_layers(value: Any, label: str) -> tuple[int, ...]:
    ok = (isinstance(value, (list, tuple))
          and all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in value)
          and all(a < b for a, b in pairwise(value)))
    if not ok:
        raise ValueError(
            f"adapter {label}: `layers` is a list of layer indices, each at least 0, "
            f"ascending and each once, such as [3, 4, 5]; got {value!r}")
    return tuple(value)


def resolve(
    value: Any,
    fetch: Callable[[str], Mapping[str, Any]],
) -> ModelRef:
    ref = parse(value)
    if ref.is_endpoint:
        return ref
    # external: mechbench-models — the ModelRef wire schema caps `adapters` at 8 (src/protocol.ts); change both together
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
        adapter_layers=ref.adapter_layers,
    )
