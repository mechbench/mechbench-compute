from __future__ import annotations

import re
from datetime import date

import pytest

from mechbench_compute.providers import pricing
from mechbench_compute.providers.features import FEATURES

ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
ROWS = [(provider, model, price) for provider, table in pricing.PRICES.items()
        if provider != "mock" for model, price in table.items()]


@pytest.mark.parametrize(("provider", "model", "price"), ROWS,
                         ids=[f"{p}/{m}" for p, m, _ in ROWS])
def test_every_row_says_where_and_when_it_was_read(provider, model, price):
    assert price.source.startswith("https://"), f"{provider}/{model} has no source page"
    assert ISO.fullmatch(price.checked), f"{provider}/{model} has no checked date"
    assert price.status in pricing.STATUSES
    for day in (price.shutdown, price.until):
        assert day is None or ISO.fullmatch(day)
    if price.status == "alias":
        assert price.note, f"{provider}/{model} is an alias and does not say of what"
    if price.until is not None and price.then is None:
        assert price.note, f"{provider}/{model} ends {price.until} and does not say what follows"


def test_every_provider_names_its_pricing_page():
    assert set(pricing.PRICES) - {"mock"} <= set(pricing.PROVIDER_PAGES)


def test_a_row_prices_its_id_its_snapshots_and_its_latest_alias_only():
    assert pricing.price_for("anthropic", "claude-haiku-4-5-20251001") is not None
    assert pricing.price_for("openai", "gpt-5-2025-08-07") is not None
    assert pricing.price_for("xai", "grok-4.5-latest") is not None
    assert pricing.price_for("openai", "gpt-5.7") is None
    assert pricing.price_for("xai", "grok-4.9") is None
    assert pricing.price_for("openai", "gpt-5.6-sol").input == 4.0
    assert pricing.price_for("openai", "gpt-5.6-terra").input == 2.0


def test_a_long_request_is_priced_whole_at_the_long_context_rate():
    short, _ = pricing.cost_usd("openai", "gpt-6-astra", {"input_tokens": 272_000})
    long, _ = pricing.cost_usd("openai", "gpt-6-astra", {"input_tokens": 272_001})
    assert short == pytest.approx(272_000 * 10.0 / 1e6)
    assert long == pytest.approx(272_001 * 20.0 / 1e6)
    at, _ = pricing.cost_usd("xai", "grok-4.7", {"input_tokens": 200_000, "output_tokens": 1_000_000})
    assert at == pytest.approx(0.2 * 4.0 + 12.0)


def test_a_promotion_gives_way_to_the_price_after_it():
    during = pricing.price_for("gemini", "gemini-3.8-flash", day=date(2026, 12, 31))
    after = pricing.price_for("gemini", "gemini-3.8-flash", day=date(2027, 1, 1))
    assert (during.input, after.input) == (0.75, 1.50)


def test_the_release_gate_names_stale_rows_ended_prices_and_shutdowns():
    assert pricing.find_table_problems(date(2026, 9, 26)) == []
    later = pricing.find_table_problems(date(2027, 1, 1))
    assert any("anthropic/claude-opus-5-5 was last checked 2026-09-26" in p for p in later)
    assert any("openai/gpt-5.6-sol: its price held until 2026-11-21" in p for p in later)
    assert any("openai/gpt-5 was to shut down 2026-12-11" in p for p in later)
    assert not any("gemini/gemini-3.8-flash: its price held" in p for p in later)


def test_features_name_only_models_the_table_prices():
    for provider, table in FEATURES.items():
        for model in table:
            assert pricing.price_for(provider, model) is not None, f"{provider}/{model}"


@pytest.mark.parametrize(("provider", "model", "price"), ROWS,
                         ids=[f"{p}/{m}" for p, m, _ in ROWS])
def test_every_priced_model_has_decided_whether_it_takes_images(provider, model, price):
    assert model in FEATURES.get(provider, {}), f"{provider}/{model} has no features row"


def test_images_go_only_where_a_page_says_so():
    from mechbench_compute.providers.features import find_features
    from mechbench_compute.support import provider_models

    assert find_features("anthropic", "claude-haiku-4-5-20251001").images
    assert find_features("openai", "gpt-4o-2024-08-06").images
    assert not find_features("openai", "gpt-3.5-turbo-0125").images
    assert not find_features("deepseek", "deepseek-v4-pro").images
    assert not find_features("openai", "gpt-unknown").images
    rows = {(m["provider"], m["model"]): m for m in provider_models()}
    assert rows[("openai", "o3-mini")]["images"] is False
    assert rows[("gemini", "gemini-3.8-flash")]["images"] is True


def test_the_check_names_what_a_provider_serves_that_the_table_does_not_say():
    from mechbench_compute.providers.table_check import ServedModel, compare

    served = [
        ServedModel("grok-4.7", rates={"input": 2.0, "output": 6.0, "cache_read": 0.5},
                    long_context_above=199_999, long_rates={"input": 4.0, "output": 12.0}),
        ServedModel("grok-4.6", rates={"input": 2.5, "output": 6.0}),
        ServedModel("grok-5", rates={"input": 3.0, "output": 9.0}),
        ServedModel("grok-imagine-video"),
    ]
    problems = compare("xai", served).problems
    assert "grok-4.6 input: the provider's API says 2.5, the table 2.0" in problems
    assert "grok-5 is served but not priced" in problems
    assert "grok-4.5 is priced but not served" in problems
    assert not any("grok-4.7" in p for p in problems)
    assert not any("imagine" in p for p in problems)
    assert not any("grok-3" in p for p in problems)


def test_the_check_says_nothing_of_a_model_left_out_on_purpose():
    from mechbench_compute.providers.table_check import EXCLUDED, ServedModel, compare

    served = [ServedModel("grok-4.20-multi-agent-0309"), ServedModel("grok-4.20-multi-agent")]
    assert not any("multi-agent" in p for p in compare("xai", served).problems)
    for provider, table in EXCLUDED.items():
        for model, reason in table.items():
            assert reason, f"{provider}/{model} is left out without a reason"
            assert pricing.price_for(provider, model) is None, f"{provider}/{model} is priced and left out"


def test_a_shut_down_model_is_unpriced_and_borrows_no_neighbour():
    assert pricing.price_for("openai", "gpt-3.5-turbo-1106") is None
    assert pricing.price_for("openai", "gpt-3.5-turbo-instruct-0914") is None
    assert pricing.price_for("openai", "gpt-3.5-turbo") is not None
    for provider, gone in pricing.SHUT_DOWN.items():
        for model in gone:
            assert model not in pricing.PRICES.get(provider, {}), f"{provider}/{model} is shut down and priced"


def test_the_check_compares_shutdown_dates():
    from mechbench_compute.providers.table_check import ServedModel, compare

    problems = compare("openai", [ServedModel("gpt-5", shutdown="2026-11-01")]).problems
    assert "gpt-5 shuts down 2026-11-01 by the provider's API; the table says 2026-12-11" in problems
