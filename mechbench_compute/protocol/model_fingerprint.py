from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Applied:
    sha256: str
    label: str | None = None
    layers: tuple[int, ...] | None = None
    scale: float | None = None

    def to_wire(self) -> dict[str, Any]:
        if self.label is None:
            return {"port": "adapter", **({"scale": self.scale} if self.scale is not None else {})}
        return {"bench": self.label, **({"layers": list(self.layers)} if self.layers is not None else {})}

    def describe(self) -> dict[str, Any]:
        return {**self.to_wire(), "sha256": self.sha256}


@dataclass
class ModelFingerprint:
    base: str
    marks: Mapping[str, Any] = field(repr=False)
    applied: list[Applied] = field(default_factory=list)
    holders: list[Any] = field(default_factory=list, repr=False)

    def read_adapters(self) -> list[dict[str, Any]]:
        return [a.to_wire() for a in self.applied if a.label is not None]

    def read_fused(self) -> list[dict[str, Any]]:
        return [a.to_wire() for a in self.applied]

    def read_changed(self, model: Any) -> list[str]:
        return read_lora(model).read_changed(model.lm, self.marks)

    def push(self, applied: Iterable[Applied]) -> None:
        self.applied.extend(applied)

    def drop(self, applied: Iterable[Applied]) -> None:
        for entry in reversed(list(applied)):
            at = max((i for i, a in enumerate(self.applied) if a == entry), default=None)
            if at is not None:
                del self.applied[at]

    def is_held_by(self, ctx: Any) -> bool:
        return any(h is ctx for h in self.holders)

    def hold(self, ctx: Any) -> None:
        self.holders.append(ctx)

    def release(self, ctx: Any) -> None:
        self.holders[:] = [h for h in self.holders if h is not ctx]

    def to_wire(self) -> dict[str, Any]:
        return {"base": self.base, "adapters": [a.describe() for a in self.applied]}


def read_applied(ref: Any, port: Any = None, scale: float | None = None) -> list[Applied]:
    labels = tuple(getattr(ref, "adapter_labels", ()) or ())
    payloads = tuple(getattr(ref, "adapter_payloads", ()) or ())
    layers = tuple(getattr(ref, "adapter_layers", ()) or ())
    out = [Applied(hash_adapter(payload), label, None if chosen is None else tuple(chosen))
           for label, payload, chosen in zip(labels, payloads, layers)]
    if port:
        out.append(Applied(hash_adapter(port), scale=scale))
    return out


def hash_adapter(payload: Any) -> str:
    from mechbench_compute.resume import content_hash

    data = payload.get("data") if isinstance(payload, Mapping) else None
    if isinstance(data, (bytes, bytearray, memoryview)):
        return content_hash({"data": hashlib.sha256(data).hexdigest(),
                             "lora": dict(payload.get("lora") or {})})
    return content_hash(payload)


def read_lora(model: Any) -> Any:
    from mechbench_compute import backends

    try:
        return backends.load_lora(backends.find(backends.backend_of(model)))
    except backends.BackendRefused:
        return None


def mark_loaded(model: Any, base: Any) -> None:
    if getattr(model, "fingerprint", None) is not None or not hasattr(model, "lm"):
        return
    mark = getattr(read_lora(model), "mark_weights", None)
    if mark is not None:
        model.fingerprint = ModelFingerprint(str(base), mark(model.lm))
