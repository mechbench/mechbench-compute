from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone

TABLE_VERSION = "2026-09-26"

PROVIDER_PAGES: dict[str, str] = {
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "openai": "https://developers.openai.com/api/docs/pricing",
    "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
    "xai": "https://docs.x.ai/developers/pricing",
    "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
    "fireworks": "https://docs.fireworks.ai/serverless/pricing",
}

STATUSES = ("current", "preview", "legacy", "deprecated", "alias")

_SNAPSHOT = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{4}|latest)")


@dataclass(frozen=True)
class LongContext:
    above: int
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    cache_write_1h: float | None = None
    long_context: LongContext | None = None
    checked: str = ""
    source: str = ""
    status: str = "current"
    shutdown: str | None = None
    until: str | None = None
    then: Price | None = None
    note: str = ""
    family: bool = False

    def read_rate(self) -> float:
        return self.input if self.cache_read is None else self.cache_read

    def write_rate(self) -> float:
        return self.input if self.cache_write is None else self.cache_write

    def write_1h_rate(self) -> float:
        return self.write_rate() if self.cache_write_1h is None else self.cache_write_1h

    def over(self, input_tokens: int) -> Price:
        lc = self.long_context
        if lc is None or input_tokens <= lc.above:
            return self
        return replace(self, input=lc.input, output=lc.output,
                       cache_read=lc.cache_read if lc.cache_read is not None else self.cache_read,
                       cache_write=lc.cache_write if lc.cache_write is not None else self.cache_write,
                       long_context=None)

    def on(self, day: date) -> Price:
        price = self
        while price.until is not None and price.then is not None and day.isoformat() > price.until:
            price = price.then
        return price


def _p(input: float, output: float, read: float | None = None, write: float | None = None,
       write_1h: float | None = None, *, provider: str, checked: str = "2026-09-26",
       source: str | None = None, **kw) -> Price:
    return Price(input, output, read, write, write_1h, checked=checked,
                 source=source or PROVIDER_PAGES[provider], **kw)


def _lc(above: int, base: tuple[float, float, float | None, float | None],
        input_x: float, output_x: float) -> LongContext:
    i, o, r, w = base
    return LongContext(above, round(i * input_x, 6), round(o * output_x, 6),
                       None if r is None else round(r * input_x, 6),
                       None if w is None else round(w * input_x, 6))


def _anthropic(i, o, r, w, w1, **kw) -> Price:
    return _p(i, o, r, w, w1, provider="anthropic", **kw)


def _openai(i, o, r=None, w=None, *, long=True, **kw) -> Price:
    return _p(i, o, r, w, provider="openai",
              long_context=_lc(272_000, (i, o, r, w), 2.0, 1.5) if long else None, **kw)


def _gemini(i, o, r, *, long: tuple[float, float, float] | None = None, **kw) -> Price:
    lc = None if long is None else LongContext(200_000, long[0], long[1], long[2])
    return _p(i, o, r, provider="gemini", long_context=lc, **kw)


def _xai(i, o, r, **kw) -> Price:
    return _p(i, o, r, provider="xai",
              long_context=_lc(199_999, (i, o, r, None), 2.0, 2.0), **kw)


_DEEPSEEK_NOTE = ("peak rate; off-peak is half, outside 01:00-04:00 and 06:00-10:00 UTC on "
                  "weekdays, and all day on weekends and Chinese public holidays")


def _deepseek(i, o, r, **kw) -> Price:
    return _p(i, o, r, provider="deepseek", note=kw.pop("note", _DEEPSEEK_NOTE), **kw)


def _fireworks(i, o, r, **kw) -> Price:
    return _p(i, o, r, provider="fireworks", note=kw.pop(
        "note", "priced one by one; Fireworks prices its other serverless models by size, "
        "which a model id does not say, so they are unpriced here"), **kw)


_GEMINI_FLASH_PROMO = dict(until="2026-12-31", then=_gemini(1.50, 7.50, 0.15),
                           note="promotional price until 2026-12-31; doubles on 2027-01-01")
_GROK_43 = (1.25, 2.50, 0.20)
_GROK_420 = (1.25, 2.50, 0.20)

PRICES: dict[str, dict[str, Price]] = {
    "anthropic": {
        "claude-fable-5-1": _anthropic(10.0, 50.0, 0.25, 12.5, 20.0, shutdown="2027-09-01"),
        "claude-opus-5-5": _anthropic(4.0, 20.0, 0.20, 5.0, 8.0, shutdown="2027-09-22"),
        "claude-sonnet-5": _anthropic(2.0, 10.0, 0.20, 2.5, 4.0, shutdown="2027-06-30"),
        "claude-haiku-4-5": _anthropic(1.0, 5.0, 0.10, 1.25, 2.0, shutdown="2026-10-15"),
        "claude-fable-5": _anthropic(10.0, 50.0, 1.0, 12.5, 20.0, status="legacy", shutdown="2027-06-09"),
        "claude-opus-5": _anthropic(5.0, 25.0, 0.5, 6.25, 10.0, status="legacy", shutdown="2027-07-24"),
        "claude-opus-4-8": _anthropic(5.0, 25.0, 0.5, 6.25, 10.0, status="legacy", shutdown="2027-05-28"),
        "claude-opus-4-7": _anthropic(5.0, 25.0, 0.5, 6.25, 10.0, status="legacy", shutdown="2027-04-16"),
        "claude-opus-4-6": _anthropic(5.0, 25.0, 0.5, 6.25, 10.0, status="legacy", shutdown="2027-02-05"),
        "claude-opus-4-5": _anthropic(5.0, 25.0, 0.5, 6.25, 10.0, status="legacy", shutdown="2026-11-24"),
        "claude-sonnet-4-6": _anthropic(3.0, 15.0, 0.3, 3.75, 6.0, status="legacy", shutdown="2027-02-17"),
    },
    "openai": {
        "gpt-6-astra": _openai(10.0, 50.0, 1.0, 12.5),
        "gpt-6-sol": _openai(2.0, 10.0, 0.2, 2.5),
        "gpt-6-luna": _openai(0.10, 0.50, 0.01, 0.125),
        "gpt-5.6": _openai(4.0, 20.0, 0.4, 5.0, status="alias", note="alias of gpt-5.6-sol",
                           until="2026-11-21"),
        "gpt-5.6-sol": _openai(4.0, 20.0, 0.4, 5.0, until="2026-11-21",
                               note="promotional price at least through 2026-11-21; check again then"),
        "gpt-5.6-terra": _openai(2.0, 12.0, 0.2, 2.5),
        "gpt-5.6-luna": _openai(0.20, 1.20, 0.02, 0.25),
        "gpt-5.5": _openai(5.0, 30.0, 0.5),
        "gpt-5.5-pro": _openai(30.0, 180.0),
        "gpt-5.4": _openai(2.5, 15.0, 0.25),
        "gpt-5.4-pro": _openai(30.0, 180.0),
        "gpt-5.4-mini": _openai(0.75, 4.5, 0.075, long=False),
        "gpt-5.4-nano": _openai(0.20, 1.25, 0.02, long=False),
        "gpt-5.2": _openai(1.75, 14.0, 0.175, long=False, status="legacy"),
        "gpt-5.2-pro": _openai(21.0, 168.0, long=False, status="legacy"),
        "gpt-5.1": _openai(1.25, 10.0, 0.125, long=False, status="legacy"),
        "gpt-5": _openai(1.25, 10.0, 0.125, long=False, status="deprecated", shutdown="2026-12-11"),
        "gpt-5-mini": _openai(0.25, 2.0, 0.025, long=False, status="deprecated", shutdown="2026-12-11"),
        "gpt-5-nano": _openai(0.05, 0.40, 0.005, long=False, status="deprecated", shutdown="2026-12-11"),
        "gpt-4.1": _openai(2.0, 8.0, 0.5, long=False, status="legacy"),
        "gpt-4.1-mini": _openai(0.40, 1.60, 0.10, long=False, status="legacy"),
        "gpt-4o": _openai(2.5, 10.0, 1.25, long=False, status="legacy"),
        "gpt-4o-mini": _openai(0.15, 0.60, 0.075, long=False, status="legacy"),
        "o3": _openai(2.0, 8.0, 0.5, long=False, status="deprecated", shutdown="2026-12-11"),
        "o4-mini": _openai(1.10, 4.40, 0.275, long=False, status="deprecated", shutdown="2026-10-23"),
    },
    "gemini": {
        "gemini-3.8-flash": _gemini(0.75, 3.75, 0.075, **_GEMINI_FLASH_PROMO),
        "gemini-3.7-flash": _gemini(0.75, 3.75, 0.075, **_GEMINI_FLASH_PROMO),
        "gemini-3.6-flash": _gemini(0.75, 3.75, 0.075, **_GEMINI_FLASH_PROMO),
        "gemini-3.5-flash": _gemini(1.50, 9.00, 0.15),
        "gemini-3.5-flash-lite": _gemini(0.30, 2.50, 0.03),
        "gemini-3.1-flash-lite": _gemini(0.25, 1.50, 0.025, status="deprecated", shutdown="2027-05-07"),
        "gemini-3.1-pro-preview": _gemini(2.0, 12.0, 0.20, long=(4.0, 18.0, 0.40), status="preview"),
        "gemini-3.1-pro-preview-customtools": _gemini(2.0, 12.0, 0.20, long=(4.0, 18.0, 0.40),
                                                      status="preview"),
        "gemini-3-flash-preview": _gemini(0.50, 3.00, 0.05, status="preview"),
        "gemini-2.5-pro": _gemini(1.25, 10.0, 0.125, long=(2.50, 15.0, 0.25), status="legacy",
                                  note="served only to projects that used it before"),
        "gemini-2.5-flash": _gemini(0.30, 2.50, 0.03, status="legacy",
                                    note="served only to projects that used it before"),
        "gemini-2.5-flash-lite": _gemini(0.10, 0.40, 0.01, status="legacy",
                                         note="served only to projects that used it before"),
    },
    "xai": {
        "grok-4.7": _xai(2.0, 6.0, 0.5),
        "grok-4.6": _xai(2.0, 6.0, 0.5),
        "grok-4.5": _xai(2.0, 6.0, 0.3),
        "grok-build-latest": _xai(2.0, 6.0, 0.3, status="alias", note="alias of grok-4.5"),
        "grok-4.3": _xai(*_GROK_43),
        "grok-4.20": _xai(*_GROK_420, status="alias", note="alias of grok-4.20-0309-reasoning"),
        "grok-4.20-reasoning": _xai(*_GROK_420, status="alias", note="alias of grok-4.20-0309-reasoning"),
        "grok-4.20-0309-reasoning": _xai(*_GROK_420),
        "grok-4.20-non-reasoning": _xai(*_GROK_420, status="alias",
                                        note="alias of grok-4.20-0309-non-reasoning"),
        "grok-4.20-0309-non-reasoning": _xai(*_GROK_420),
        "grok-build-0.1": _xai(1.0, 2.0, 0.2),
        "grok-code-fast-1": _xai(1.0, 2.0, 0.2, status="alias", note="alias of grok-build-0.1"),
        "grok-code-fast": _xai(1.0, 2.0, 0.2, status="alias", note="alias of grok-build-0.1"),
        "grok-4-0709": _xai(*_GROK_43, status="alias",
                            note="retired 2026-05-15; served and billed as grok-4.3"),
        "grok-3": _xai(*_GROK_43, status="alias",
                       note="retired 2026-05-15; served and billed as grok-4.3"),
    },
    "deepseek": {
        "deepseek-flash": _deepseek(0.30, 1.20, 0.006),
        "deepseek-v4-pro": _deepseek(1.32, 3.96, 0.044,
                                     note=_DEEPSEEK_NOTE + "; DeepSeek's news post of 2026-09-10 said "
                                     "Pro would be served as Flash from 2026-09-14, its changelog "
                                     "and pricing page say it stays"),
        "deepseek-v4-flash": _deepseek(0.30, 1.20, 0.006, status="alias",
                                       note="retired model name, served and billed as deepseek-flash"),
        "deepseek-v4-flash-vision-exp": _deepseek(0.30, 1.20, 0.006, status="alias",
                                                  note="served and billed as deepseek-flash"),
    },
    "fireworks": {
        "accounts/fireworks/models/qwen3p8-max": _fireworks(2.00, 6.00, 0.25),
        "accounts/fireworks/models/kimi-k3": _fireworks(3.00, 15.00, 0.30),
        "accounts/fireworks/models/deepseek-v4p1-flash": _fireworks(0.30, 1.20, 0.006),
        "accounts/fireworks/models/gpt-oss-120b": _fireworks(0.15, 0.60, 0.015),
        "accounts/fireworks/models/glm-5p3-flash": _fireworks(0.15, 0.50, 0.03),
    },
    "mock": {
        "": Price(1.0, 3.0, 0.1, 1.25, family=True, checked="n/a", source="n/a"),
    },
}


def matches(key: str, model: str, family: bool) -> bool:
    if family:
        return model.startswith(key)
    return model == key or (model.startswith(key) and _SNAPSHOT.fullmatch(model[len(key):]) is not None)


def price_for(provider: str, model: str, *, day: date | None = None) -> Price | None:
    table = PRICES.get(provider)
    if not table:
        return None
    best: tuple[int, Price] | None = None
    for key, price in table.items():
        if matches(key, model, price.family) and (best is None or len(key) > best[0]):
            best = (len(key), price)
    if best is None:
        return None
    return best[1].on(day or datetime.now(timezone.utc).date())


MAX_AGE_DAYS = 60


def find_table_problems(day: date, max_age_days: int = MAX_AGE_DAYS) -> list[str]:
    out: list[str] = []
    for provider, table in PRICES.items():
        if provider == "mock":
            continue
        for model, price in table.items():
            name = f"{provider}/{model}"
            if (day - date.fromisoformat(price.checked)).days > max_age_days:
                out.append(f"{name} was last checked {price.checked}, over {max_age_days} days ago")
            if price.until is not None and price.then is None and day.isoformat() > price.until:
                out.append(f"{name}: its price held until {price.until}; check what it is now")
            if price.shutdown is not None and day.isoformat() > price.shutdown:
                out.append(f"{name} was to shut down {price.shutdown}; remove it, or mark it an alias")
    return out


def cost_usd(provider: str, model: str, usage: Mapping[str, int], *,
             day: date | None = None) -> tuple[float, bool]:
    found = price_for(provider, model, day=day)
    if found is None:
        return 0.0, False
    price = found.over(int(usage.get("input_tokens", 0) or 0))
    read = int(usage.get("cache_read_tokens", 0) or 0)
    write = int(usage.get("cache_write_tokens", 0) or 0)
    write_1h = min(write, int(usage.get("cache_write_1h_tokens", 0) or 0))
    plain = max(0, int(usage.get("input_tokens", 0) or 0) - read - write)
    out = int(usage.get("output_tokens", 0) or 0)
    total = (plain * price.input + read * price.read_rate()
             + (write - write_1h) * price.write_rate()
             + write_1h * price.write_1h_rate() + out * price.output) / 1_000_000
    return round(total, 8), True


def worst_case_usd(provider: str, model: str, *, input_tokens: int,
                   max_tokens: int) -> tuple[float, bool]:
    return cost_usd(provider, model, {"input_tokens": input_tokens,
                                      "output_tokens": max_tokens})
