"""The model algebra's reference form (task 000312, Arc A).

A ``$model`` may be more than an HF repo: the full grammar is a *base*
plus an ordered *adapter stack*::

    ModelRef  := { base: Base, adapters: [Adapter, ...] }
    Base      := {"hf": "repo[@revision]"} | {"bench": "<label>"}
    Adapter   := {"bench": "<label>"}

A bare string ``"repo[@revision]"`` remains valid forever and reads as
``{base: {hf: ...}, adapters: []}`` — every protocol written before
this module keeps meaning what it meant.

Base and adapter sources are EXPLICIT. An HF repo and a bench label are
both slash-paths, so telling them apart by shape would be a guess, and
a guess here would load the wrong weights.

The recursion the epic describes flattens by construction: fine-tuning
on {B, [a1]} yields a2 and the product is {B, [a1, a2]}; a merge
collapses a stack into a checkpoint that stands as a fresh Base. This
module therefore never sees a tree.

Stacks of any practical depth resolve since Arc B — the executor
fuses them in order (see lora.fuse_adapter_stack). Checkpoint bases
(Arc C) parse fine and fail loudly at *load* time naming their arc: a
reference that parses but cannot yet run should say so in the run's
error, not in a stack trace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelRef:
    """A parsed reference. `adapter_payloads` holds fetched adapter
    objects (safetensors bytes + lora config), aligned with
    `adapter_labels`."""

    base_kind: str  # "hf" | "bench"
    base: str
    adapter_labels: tuple[str, ...] = ()
    adapter_payloads: tuple[Mapping[str, Any], ...] = field(default=(), compare=False)

    def describe(self) -> str:
        """`hf:repo@rev (+2 adapters)` — for telemetry and manifests."""
        tail = ""
        if self.adapter_labels:
            n = len(self.adapter_labels)
            tail = f" (+{n} adapter{'s' if n != 1 else ''})"
        return f"{self.base_kind}:{self.base}{tail}"


def parse(value: Any) -> ModelRef:
    """String or structured form -> ModelRef. Raises ValueError with a
    sentence, never a shrug — this runs at the top of a job."""
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
    """Parse and fetch the adapter payloads.

    `fetch` is injected (bench.fetch in production) so tests never
    touch the network — the runner-conftest fence rule, applied here
    from the start.
    """
    ref = parse(value)
    if ref.base_kind == "bench":
        raise NotImplementedError(
            "checkpoint bases (a merged model stored as a bench object) "
            "arrive with task 000312 Arc C; today a base must be an HF "
            "repo[@revision]"
        )
    if len(ref.adapter_labels) > 8:
        # Mirrors the wire schema's cap. Linear fuse cost makes very
        # deep stacks a smell anyway — merge (Arc C) is the pressure
        # valve.
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
