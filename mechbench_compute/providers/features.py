from __future__ import annotations

from dataclasses import dataclass

PROMPT_CACHES = ("5m", "1h")


@dataclass(frozen=True)
class Features:
    effort: tuple[str, ...] = ()
    reasoning_displays: tuple[str, ...] = ()
    prompt_cache: bool = False


_CLAUDE_EFFORT = ("low", "medium", "high", "xhigh", "max")

FEATURES: dict[str, dict[str, Features]] = {
    "anthropic": {
        "claude-opus-5-5": Features(_CLAUDE_EFFORT, ("summarized", "updates"), True),
        "claude-fable-5-1": Features(_CLAUDE_EFFORT, ("summarized", "updates"), True),
        "claude-fable-5": Features(_CLAUDE_EFFORT, ("summarized", "updates"), True),
        "claude-opus-5": Features(_CLAUDE_EFFORT, ("summarized",), True),
        "claude-sonnet-5": Features(_CLAUDE_EFFORT, ("summarized",), True),
        "claude-haiku-4-5": Features((), (), True),
    },
    "openai": {
        "gpt-5": Features(("minimal", "low", "medium", "high")),
    },
}


def find_features(provider: str, model: str) -> Features:
    table = FEATURES.get(provider) or {}
    best: tuple[int, Features] | None = None
    for prefix, feats in table.items():
        if model.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), feats)
    return best[1] if best else Features()
