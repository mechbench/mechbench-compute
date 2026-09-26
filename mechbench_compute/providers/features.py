from __future__ import annotations

from dataclasses import dataclass

PROMPT_CACHES = ("5m", "1h")


@dataclass(frozen=True)
class Features:
    effort: tuple[str, ...] = ()
    reasoning_displays: tuple[str, ...] = ()
    prompt_cache: bool = False


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
        "claude-fable-5-1": Features(_CLAUDE_EFFORT, _UPDATES, True),
        "claude-opus-5-5": Features(_CLAUDE_EFFORT, _UPDATES, True),
        "claude-fable-5": Features(_CLAUDE_EFFORT, _UPDATES, True),
        "claude-sonnet-5": Features(_CLAUDE_EFFORT, _SUMMARIZED, True),
        "claude-opus-5": Features(_CLAUDE_EFFORT, _SUMMARIZED, True),
        "claude-opus-4-8": Features(_CLAUDE_EFFORT, _SUMMARIZED, True),
        "claude-opus-4-7": Features(_CLAUDE_EFFORT, _SUMMARIZED, True),
        "claude-opus-4-6": Features(_CLAUDE_4_6_EFFORT, _SUMMARIZED, True),
        "claude-sonnet-4-6": Features(_CLAUDE_4_6_EFFORT, _SUMMARIZED, True),
        "claude-opus-4-5": Features((), (), True),
        "claude-haiku-4-5": Features((), (), True),
    },
    "openai": {
        "gpt-6-astra": Features(("low", "medium", "high", "xhigh", "max")),
        "gpt-6-sol": Features(_OPENAI_6),
        "gpt-6-luna": Features(_OPENAI_6),
        "gpt-5.6": Features(_OPENAI_6),
        "gpt-5.6-sol": Features(_OPENAI_6),
        "gpt-5.6-terra": Features(_OPENAI_6),
        "gpt-5.6-luna": Features(_OPENAI_6),
        "gpt-5.5": Features(_OPENAI_5_4),
        "gpt-5.5-pro": Features(("medium", "high", "xhigh")),
        "gpt-5.4": Features(_OPENAI_5_4),
        "gpt-5.4-mini": Features(_OPENAI_5_4),
        "gpt-5.4-nano": Features(_OPENAI_5_4),
        "gpt-5.2": Features(_OPENAI_5_4),
        "gpt-5.1": Features(("none", "low", "medium", "high")),
        "gpt-5": Features(("minimal", "low", "medium", "high")),
    },
    "xai": {
        "grok-4.7": Features(_GROK_47),
        "grok-4.6": Features(_GROK_47),
        "grok-4.5": Features(("low", "medium", "high")),
        "grok-4.3": Features(("none", "low", "medium", "high")),
    },
    "deepseek": {
        "deepseek-flash": Features(_DEEPSEEK),
        "deepseek-v4-pro": Features(_DEEPSEEK),
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
