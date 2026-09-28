from __future__ import annotations

from dataclasses import dataclass

PROMPT_CACHES = ("5m", "1h")


@dataclass(frozen=True)
class Features:
    effort: tuple[str, ...] = ()
    reasoning_displays: tuple[str, ...] = ()
    prompt_cache: bool = False
    images: bool = False


_CLAUDE_EFFORT = ("low", "medium", "high", "xhigh", "max")
_CLAUDE_4_6_EFFORT = ("low", "medium", "high", "max")
_UPDATES = ("summarized", "updates")
_SUMMARIZED = ("summarized",)
_OPENAI_6 = ("none", "low", "medium", "high", "xhigh", "max")
_OPENAI_5_4 = ("none", "low", "medium", "high", "xhigh")
_GROK_47 = ("low", "medium", "high", "xhigh")
_DEEPSEEK = ("low", "high", "max")

FEATURES: dict[str, dict[str, Features]] = {
    "anthropic": {
        "claude-fable-5-1": Features(_CLAUDE_EFFORT, _UPDATES, True, images=True),
        "claude-opus-5-5": Features(_CLAUDE_EFFORT, _UPDATES, True, images=True),
        "claude-fable-5": Features(_CLAUDE_EFFORT, _UPDATES, True, images=True),
        "claude-sonnet-5": Features(_CLAUDE_EFFORT, _SUMMARIZED, True, images=True),
        "claude-opus-5": Features(_CLAUDE_EFFORT, _SUMMARIZED, True, images=True),
        "claude-opus-4-8": Features(_CLAUDE_EFFORT, _SUMMARIZED, True, images=True),
        "claude-opus-4-7": Features(_CLAUDE_EFFORT, _SUMMARIZED, True, images=True),
        "claude-opus-4-6": Features(_CLAUDE_4_6_EFFORT, _SUMMARIZED, True, images=True),
        "claude-sonnet-4-6": Features(_CLAUDE_4_6_EFFORT, _SUMMARIZED, True, images=True),
        "claude-opus-4-5": Features((), (), True, images=True),
        "claude-haiku-4-5": Features((), (), True, images=True),
        "claude-sonnet-4-5": Features(images=True),
    },
    "openai": {
        "gpt-6-astra": Features(("low", "medium", "high", "xhigh", "max"), images=True),
        "gpt-6-sol": Features(_OPENAI_6, images=True),
        "gpt-6-luna": Features(_OPENAI_6, images=True),
        "gpt-5.6": Features(_OPENAI_6, images=True),
        "gpt-5.6-sol": Features(_OPENAI_6, images=True),
        "gpt-5.6-terra": Features(_OPENAI_6, images=True),
        "gpt-5.6-luna": Features(_OPENAI_6, images=True),
        "gpt-5.5": Features(_OPENAI_5_4, images=True),
        "gpt-5.5-pro": Features(("medium", "high", "xhigh"), images=True),
        "gpt-5.4": Features(_OPENAI_5_4, images=True),
        "gpt-5.4-pro": Features(images=True),
        "gpt-5.4-mini": Features(_OPENAI_5_4, images=True),
        "gpt-5.4-nano": Features(_OPENAI_5_4, images=True),
        "gpt-5.3-codex": Features(images=True),
        "gpt-5.2": Features(_OPENAI_5_4, images=True),
        "gpt-5.2-pro": Features(images=True),
        "gpt-5.1": Features(("none", "low", "medium", "high"), images=True),
        "gpt-5": Features(("minimal", "low", "medium", "high"), images=True),
        "gpt-5-pro": Features(images=True),
        "gpt-5-mini": Features(images=True),
        "gpt-5-nano": Features(images=True),
        "gpt-4.1": Features(images=True),
        "gpt-4.1-mini": Features(images=True),
        "gpt-4.1-nano": Features(images=True),
        "gpt-4o": Features(images=True),
        "gpt-4o-mini": Features(images=True),
        "gpt-4o-2024-05-13": Features(images=True),
        "gpt-4-turbo": Features(images=True),
        "gpt-4": Features(),
        "gpt-4-0613": Features(),
        "gpt-3.5-turbo": Features(),
        "gpt-3.5-turbo-0125": Features(),
        "gpt-3.5-turbo-1106": Features(),
        "gpt-3.5-turbo-instruct": Features(),
        "o1": Features(images=True),
        "o1-pro": Features(images=True),
        "o3": Features(images=True),
        "o3-pro": Features(images=True),
        "o3-mini": Features(),
        "o4-mini": Features(images=True),
    },
    "gemini": {
        "gemini-3.8-flash": Features(images=True),
        "gemini-3.7-flash": Features(images=True),
        "gemini-3.6-flash": Features(images=True),
        "gemini-3.5-flash": Features(images=True),
        "gemini-3.5-flash-lite": Features(images=True),
        "gemini-3.1-flash-lite": Features(images=True),
        "gemini-3.1-pro-preview": Features(images=True),
        "gemini-3.1-pro-preview-customtools": Features(images=True),
        "gemini-3-flash-preview": Features(images=True),
        "gemini-2.5-pro": Features(images=True),
        "gemini-2.5-flash": Features(images=True),
        "gemini-2.5-flash-lite": Features(images=True),
    },
    "xai": {
        "grok-4.7": Features(_GROK_47, images=True),
        "grok-4.6": Features(_GROK_47, images=True),
        "grok-4.5": Features(_GROK_47, images=True),
        "grok-build-latest": Features(images=True),
        "grok-4.3": Features(("none", "low", "medium", "high"), images=True),
        "grok-4.20": Features(images=True),
        "grok-4.20-reasoning": Features(images=True),
        "grok-4.20-0309-reasoning": Features(images=True),
        "grok-4.20-non-reasoning": Features(images=True),
        "grok-4.20-0309-non-reasoning": Features(images=True),
        "grok-build-0.1": Features(images=True),
        "grok-code-fast-1": Features(images=True),
        "grok-code-fast": Features(images=True),
        "grok-4-0709": Features(),
        "grok-3": Features(),
    },
    "deepseek": {
        "deepseek-flash": Features(_DEEPSEEK, images=True),
        "deepseek-v4-pro": Features(_DEEPSEEK),
        "deepseek-v4-flash": Features(images=True),
        "deepseek-v4-flash-vision-exp": Features(images=True),
    },
    "fireworks": {
        "accounts/fireworks/models/qwen3p8-max": Features(images=True),
        "accounts/fireworks/models/kimi-k3": Features(images=True),
        "accounts/fireworks/models/deepseek-v4p1-flash": Features(images=True),
        "accounts/fireworks/models/gpt-oss-120b": Features(),
        "accounts/fireworks/models/glm-5p3-flash": Features(images=True),
    },
}


def find_features(provider: str, model: str) -> Features:
    from mechbench_compute.providers.pricing import matches

    table = FEATURES.get(provider) or {}
    best: tuple[int, Features] | None = None
    for key, feats in table.items():
        if matches(key, model, False) and (best is None or len(key) > best[0]):
            best = (len(key), feats)
    return best[1] if best else Features()
