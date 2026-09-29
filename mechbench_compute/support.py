from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._arch import GLOBAL_HOOK_POINTS, LAYER_HOOK_POINTS

LEVELS: dict[str, str] = {
    "full": "every hook point: the residual stream, attention and MLP internals, "
            "the per-layer gate, the embedding, the final norm and the logits",
    "core": "the residual stream, attention and MLP outputs, attention weights, "
            "per-head outputs, q/k/v, the embedding, the final norm and the logits; "
            "no MLP internals and no pre-norm or pre-rope points",
    "text": "generation, scoring and training, with no hook points",
}

LOADERS: tuple[str, ...] = ("mlx-vlm", "mlx-lm")

CORE_LAYER_POINTS: tuple[str, ...] = (
    "resid_pre", "attn_out", "mlp_out", "resid_post",
    "attn.weights", "attn.per_head_out", "attn.q", "attn.k", "attn.v",
)
CORE_GLOBAL_POINTS: tuple[str, ...] = GLOBAL_HOOK_POINTS


@dataclass(frozen=True)
class Refusal:
    config_key: str
    reason: str


@dataclass(frozen=True)
class Unembed:
    norm: Any
    project: Callable[[Any], Any]
    softcap: float | None = None


@dataclass(frozen=True)
class Architecture:
    model_type: str
    name: str
    loader: str
    generate: bool
    score: bool
    train: bool
    layer_points: tuple[str, ...]
    global_points: tuple[str, ...]
    residual_law: str
    load: Callable[..., tuple[Any, Any]]
    arch_of: Callable[..., Any]
    forward: Callable[..., tuple[Any, Any]]
    lm: Callable[[Any], Any]
    prompt_cache: Callable[[Any], Any]
    head_logits: Callable[[Any, Any], Any]
    project_to_logits: Callable[[Any, Any], Any]
    tokenize: Callable[..., Any]
    attribution_unembed: Callable[[Any], Unembed]
    head_weights: Callable[[Any, int, int], Any]
    dialect: Any
    reasoning: tuple[Any, ...]
    adapter_keys: Any
    refused_when: tuple[Refusal, ...] = ()
    config_defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> str:
        if not self.layer_points and not self.global_points:
            return "text"
        if (set(self.layer_points) == set(LAYER_HOOK_POINTS)
                and set(self.global_points) == set(GLOBAL_HOOK_POINTS)):
            return "full"
        return "core"

    def supports(self, point: str, *, layer_scoped: bool) -> bool:
        return point in (self.layer_points if layer_scoped else self.global_points)


def refusal(config: Mapping[str, Any]) -> str | None:
    from .architectures import ARCHITECTURES, for_type

    model_type = str(config.get("model_type") or "").lower()
    arch = for_type(model_type)
    if arch is None:
        return (f"model_type {model_type!r} is not one compute loads; it loads "
                f"{', '.join(a.model_type for a in ARCHITECTURES)}")
    text = config.get("text_config")
    scopes = [config] + ([text] if isinstance(text, Mapping) else [])
    for r in arch.refused_when:
        if any(scope.get(r.config_key) for scope in scopes):
            return r.reason
    return None


def local_architectures() -> list[dict[str, Any]]:
    from .architectures import ARCHITECTURES

    return [{
        "modelType": a.model_type,
        "name": a.name,
        "loader": a.loader,
        "level": a.level,
        "generate": a.generate,
        "score": a.score,
        "train": a.train,
        "layerPoints": list(a.layer_points),
        "globalPoints": list(a.global_points),
        "refusedWhen": [{"configKey": r.config_key, "reason": r.reason}
                        for r in a.refused_when],
        "configDefaults": dict(a.config_defaults),
    } for a in ARCHITECTURES]


def provider_models() -> list[dict[str, Any]]:
    from .providers import pricing
    from .providers.features import find_features
    from .providers.registry import registry

    def rates(price: pricing.Price) -> dict[str, Any]:
        lc = price.long_context
        return {
            "inputPerMillion": price.input,
            "outputPerMillion": price.output,
            "cacheReadPerMillion": price.cache_read,
            "cacheWritePerMillion": price.cache_write,
            "cacheWrite1hPerMillion": price.cache_write_1h,
            "longContext": None if lc is None else {
                "above": lc.above,
                "inputPerMillion": lc.input,
                "outputPerMillion": lc.output,
                "cacheReadPerMillion": lc.cache_read,
                "cacheWritePerMillion": lc.cache_write,
            },
        }

    out: list[dict[str, Any]] = []
    for name, spec in registry().items():
        if not spec.base_url:
            continue
        for model, price in pricing.PRICES.get(name, {}).items():
            out.append({
                "provider": name,
                "model": model,
                **rates(price),
                "until": price.until,
                "then": None if price.then is None else rates(price.then),
                "status": price.status,
                "shutdown": price.shutdown,
                "note": price.note,
                "source": price.source,
                "checked": price.checked,
                "effortLevels": list(find_features(name, model).effort),
                "reasoningDisplays": list(find_features(name, model).reasoning_displays),
                "promptCache": find_features(name, model).prompt_cache,
                "images": find_features(name, model).images,
                "reasoning": spec.capabilities.reasoning,
                "tools": spec.capabilities.tools,
            })
    return out
