from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._arch import GLOBAL_HOOK_POINTS, LAYER_HOOK_POINTS

LEVELS: dict[str, str] = {
    "full": "every hook point: the residual stream, attention and MLP internals, "
            "the per-layer gate, the embedding, the final norm and the logits",
    "core": "the residual stream, attention and MLP outputs, attention weights, "
            "per-head outputs and q/k/v; no MLP internals, no pre-norm or pre-rope "
            "points, no embedding, final norm or logits hooks",
    "text": "generation, scoring and training, with no hook points",
}

LOADERS: tuple[str, ...] = ("mlx-vlm", "mlx-lm")

CORE_LAYER_POINTS: tuple[str, ...] = (
    "resid_pre", "attn_out", "mlp_out", "resid_post",
    "attn.weights", "attn.per_head_out", "attn.q", "attn.k", "attn.v",
)
CORE_GLOBAL_POINTS: tuple[str, ...] = ("final_norm.scale",)


@dataclass(frozen=True)
class Refusal:
    config_key: str
    reason: str


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
    refused_when: tuple[Refusal, ...] = ()

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


ARCHITECTURES: tuple[Architecture, ...] = (
    Architecture(
        model_type="gemma4", name="Gemma 4", loader="mlx-vlm",
        generate=True, score=True, train=True,
        layer_points=LAYER_HOOK_POINTS, global_points=GLOBAL_HOOK_POINTS,
        refused_when=(Refusal(
            "enable_moe_block",
            "mixture-of-experts layers (Gemma 4 26B A4B) are not wired into the "
            "gemma4 forward; only the dense checkpoints load"),),
    ),
    Architecture(
        model_type="gemma3", name="Gemma 3", loader="mlx-vlm",
        generate=True, score=True, train=True,
        layer_points=CORE_LAYER_POINTS, global_points=CORE_GLOBAL_POINTS,
    ),
    Architecture(
        model_type="qwen2", name="Qwen 2", loader="mlx-lm",
        generate=True, score=True, train=True,
        layer_points=CORE_LAYER_POINTS, global_points=CORE_GLOBAL_POINTS,
    ),
    Architecture(
        model_type="llama", name="Llama", loader="mlx-lm",
        generate=True, score=True, train=True,
        layer_points=CORE_LAYER_POINTS, global_points=CORE_GLOBAL_POINTS,
    ),
)

BY_MODEL_TYPE: dict[str, Architecture] = {a.model_type: a for a in ARCHITECTURES}

MODEL_TYPES: tuple[str, ...] = tuple(a.model_type for a in ARCHITECTURES)

MLX_LM_MODEL_TYPES: frozenset[str] = frozenset(
    a.model_type for a in ARCHITECTURES if a.loader == "mlx-lm")


def architecture(model_type: str) -> Architecture | None:
    return BY_MODEL_TYPE.get((model_type or "").lower())


def refusal(config: Mapping[str, Any]) -> str | None:
    model_type = str(config.get("model_type") or "").lower()
    arch = architecture(model_type)
    if arch is None:
        return (f"model_type {model_type!r} is not one compute loads; it loads "
                f"{', '.join(MODEL_TYPES)}")
    text = config.get("text_config")
    scopes = [config] + ([text] if isinstance(text, Mapping) else [])
    for r in arch.refused_when:
        if any(scope.get(r.config_key) for scope in scopes):
            return r.reason
    return None


def local_architectures() -> list[dict[str, Any]]:
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
                "reasoning": spec.capabilities.reasoning,
                "tools": spec.capabilities.tools,
            })
    return out
