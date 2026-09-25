from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

TABLE_VERSION = "2026-09-25"


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def read_rate(self) -> float:
        return self.input if self.cache_read is None else self.cache_read

    def write_rate(self) -> float:
        return self.input if self.cache_write is None else self.cache_write


PRICES: dict[str, dict[str, Price]] = {
    "anthropic": {
        "claude-opus-5": Price(5.0, 25.0, 0.5, 6.25),
        "claude-sonnet-5": Price(3.0, 15.0, 0.3, 3.75),
        "claude-haiku-4-5": Price(1.0, 5.0, 0.1, 1.25),
        "claude-fable-5": Price(10.0, 50.0, 1.0, 12.5),
    },
    "openai": {
        # external: OpenAI — GPT-6 Astra charges 2x input and 1.5x output on a request over 272K input tokens; this is the standard rate
        "gpt-6-astra": Price(10.0, 50.0, 1.0),
        "gpt-5": Price(1.25, 10.0, 0.125),
        "gpt-4.1": Price(2.0, 8.0, 0.5),
        "gpt-4o": Price(2.5, 10.0, 1.25),
        "o3": Price(2.0, 8.0, 0.5),
    },
    "gemini": {
        "gemini-2.5-pro": Price(1.25, 10.0, 0.31),
        "gemini-2.5-flash": Price(0.30, 2.50, 0.075),
    },
    # external: xAI — every rate doubles from 200K tokens; these are the standard rates
    "xai": {
        "grok-4.7": Price(2.0, 6.0, 0.5),
        "grok-4.6": Price(2.0, 6.0, 0.5),
        "grok-4.5": Price(2.0, 6.0, 0.3),
        "grok-4": Price(3.0, 15.0, 0.75),
        "grok-3": Price(3.0, 15.0),
    },
    # external: DeepSeek — half price off-peak; these are the peak rates
    "deepseek": {
        "deepseek-flash": Price(0.30, 1.20, 0.006),
        "deepseek-v4-flash": Price(0.30, 1.20, 0.006),
        "deepseek-v4-pro": Price(1.32, 3.96, 0.044),
    },
    "fireworks": {
        "accounts/fireworks/models/llama": Price(0.9, 0.9),
        "accounts/fireworks/models/qwen": Price(0.9, 0.9),
    },
    "mock": {
        "": Price(1.0, 3.0, 0.1, 1.25),
    },
}


def price_for(provider: str, model: str) -> Price | None:
    table = PRICES.get(provider)
    if not table:
        return None
    best: tuple[int, Price] | None = None
    for prefix, price in table.items():
        if model.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), price)
    return best[1] if best else None


def cost_usd(provider: str, model: str, usage: Mapping[str, int]) -> tuple[float, bool]:
    price = price_for(provider, model)
    if price is None:
        return 0.0, False
    read = int(usage.get("cache_read_tokens", 0) or 0)
    write = int(usage.get("cache_write_tokens", 0) or 0)
    plain = max(0, int(usage.get("input_tokens", 0) or 0) - read - write)
    out = int(usage.get("output_tokens", 0) or 0)
    total = (plain * price.input + read * price.read_rate()
             + write * price.write_rate() + out * price.output) / 1_000_000
    return round(total, 8), True


def worst_case_usd(provider: str, model: str, *, input_tokens: int,
                   max_tokens: int) -> tuple[float, bool]:
    return cost_usd(provider, model, {"input_tokens": input_tokens,
                                      "output_tokens": max_tokens})
