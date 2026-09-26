from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mechbench_compute.providers import pricing

TEXT_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude-",),
    "openai": ("gpt-", "o1", "o3", "o4"),
    "gemini": ("gemini-",),
    "xai": ("grok-",),
    "deepseek": ("deepseek-",),
}

NOT_TEXT_MARKERS = ("audio", "realtime", "image", "tts", "transcribe", "embedding", "search",
                    "live", "veo", "imagine", "moderation", "computer-use", "robotics", "cyber")


@dataclass(frozen=True)
class ServedModel:
    id: str
    aliases: tuple[str, ...] = ()
    shutdown: str | None = None
    rates: dict[str, float] | None = None
    long_context_above: int | None = None
    long_rates: dict[str, float] | None = None


@dataclass
class Findings:
    provider: str
    problems: list[str] = field(default_factory=list)


def is_text_model(provider: str, model_id: str) -> bool:
    prefixes = TEXT_MODEL_PREFIXES.get(provider, ())
    return model_id.startswith(prefixes) and not any(m in model_id for m in NOT_TEXT_MARKERS)


def compare(provider: str, served: Sequence[ServedModel]) -> Findings:
    found = Findings(provider)
    names = {m.id for m in served} | {a for m in served for a in m.aliases}
    table = pricing.PRICES.get(provider, {})
    for key, price in table.items():
        if price.family or price.status == "alias":
            continue
        if not any(pricing.matches(key, name, False) for name in names):
            found.problems.append(f"{key} is priced but not served")
    for model in served:
        if not is_text_model(provider, model.id):
            continue
        price = pricing.price_for(provider, model.id)
        if price is None:
            found.problems.append(f"{model.id} is served but not priced")
            continue
        if model.shutdown is not None and model.shutdown != price.shutdown:
            found.problems.append(
                f"{model.id} shuts down {model.shutdown} by the provider's API; the table says "
                f"{price.shutdown or 'nothing'}")
        if model.rates is not None:
            ours = {"input": price.input, "output": price.output, "cache_read": price.cache_read}
            for name, theirs in model.rates.items():
                if ours.get(name) is not None and abs(ours[name] - theirs) > 1e-9:
                    found.problems.append(
                        f"{model.id} {name}: the provider's API says {theirs}, the table {ours[name]}")
        if model.long_context_above and model.long_rates is not None:
            lc = price.long_context
            if lc is None:
                found.problems.append(
                    f"{model.id} has a long-context price above {model.long_context_above} tokens; "
                    "the table has none")
            else:
                ours_long = {"input": lc.input, "output": lc.output, "cache_read": lc.cache_read}
                for name, theirs in model.long_rates.items():
                    if ours_long.get(name) is not None and abs(ours_long[name] - theirs) > 1e-9:
                        found.problems.append(
                            f"{model.id} long-context {name}: the provider's API says {theirs}, "
                            f"the table {ours_long[name]}")
    return found
