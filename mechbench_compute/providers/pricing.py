"""Prices as data (task 000337).

A cost that a run reports must be reproducible from what the run
recorded, so prices are a versioned table with an `as_of` date, matched
by longest model prefix, and every provenance record carries the table
version that priced it. When a price changes, old runs keep costing
what they cost.

USD per MILLION tokens. `cache_read` / `cache_write` are the
prompt-caching rates where a provider offers them (Anthropic's 0.1x
read / 1.25x write, OpenAI's 0.5x read); absent, they fall back to the
input rate.

Unknown models price at 0 and are FLAGGED (`priced: false` in the call
record) rather than guessed: a made-up number in a budget is worse
than a visible hole, and the budget still counts tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

TABLE_VERSION = "2026-09-08"


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


#: (provider, model prefix) -> Price. Longest matching prefix wins, so
#: "claude-opus-5" and "claude-opus-5-1-20260115" price the same until
#: a more specific row says otherwise.
PRICES: dict[str, dict[str, Price]] = {
    "anthropic": {
        "claude-opus-5": Price(15.0, 75.0, 1.5, 18.75),
        "claude-sonnet-5": Price(3.0, 15.0, 0.3, 3.75),
        "claude-haiku-4-5": Price(1.0, 5.0, 0.1, 1.25),
        "claude-fable-5": Price(3.0, 15.0, 0.3, 3.75),
    },
    "openai": {
        "gpt-5": Price(1.25, 10.0, 0.125),
        "gpt-4.1": Price(2.0, 8.0, 0.5),
        "gpt-4o": Price(2.5, 10.0, 1.25),
        "o3": Price(2.0, 8.0, 0.5),
    },
    "gemini": {
        "gemini-2.5-pro": Price(1.25, 10.0, 0.31),
        "gemini-2.5-flash": Price(0.30, 2.50, 0.075),
    },
    "xai": {
        "grok-4": Price(3.0, 15.0, 0.75),
        "grok-3": Price(3.0, 15.0),
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
    """Longest-prefix match within a provider; None when unknown."""
    table = PRICES.get(provider)
    if not table:
        return None
    best: tuple[int, Price] | None = None
    for prefix, price in table.items():
        if model.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), price)
    return best[1] if best else None


def cost_usd(provider: str, model: str, usage: Mapping[str, int]) -> tuple[float, bool]:
    """(cost, priced). `usage` carries input_tokens, output_tokens and
    optionally cache_read_tokens / cache_write_tokens — cached input is
    priced at its own rate and NOT double-counted as plain input."""
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
    """What this call could cost if the model writes to its limit —
    the number a budget check refuses on, BEFORE the call is made."""
    return cost_usd(provider, model, {"input_tokens": input_tokens,
                                      "output_tokens": max_tokens})
