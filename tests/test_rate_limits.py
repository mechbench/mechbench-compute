"""The provider registry and the rate-limit model (task 000344).

Bucket arithmetic runs on a fake clock; the concurrency cap runs on
real threads, because a standing count is exactly the thing a fake
clock cannot test.
"""

from __future__ import annotations

import threading

import pytest

from mechbench_compute.providers import PROVIDERS
from mechbench_compute.providers import registry as reg
from mechbench_compute.providers.limiter import RateLimits
from mechbench_compute.providers.messages import request
from mechbench_compute.providers.mock import MockTransport, rate_limited


class FakeClock:
    """Time moves only when something sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 6))
        self.t += seconds


def limiter(clock: FakeClock, **limits) -> reg.TokenBucketLimiter:
    return reg.TokenBucketLimiter(
        sleep=clock.sleep, clock=clock.now,
        limits=lambda _p, _m="": reg.Limits(**limits))


class TestRegistry:
    def test_every_provider_has_a_spec_with_limits_and_capabilities(self):
        for name in PROVIDERS:
            spec = reg.spec_for(name)
            assert spec.adapter and isinstance(spec.limits, reg.Limits)
            assert spec.to_wire()["registry_version"] == reg.REGISTRY_VERSION
        with pytest.raises(ValueError, match="unknown provider"):
            reg.spec_for("altavista")

    def test_a_model_family_overrides_only_what_it_names(self):
        base = reg.limits_for("anthropic", "claude-opus-5")
        haiku = reg.limits_for("anthropic", "claude-haiku-4-5")
        assert haiku.input_tokens == 50_000 and base.input_tokens == 30_000
        # concurrency was not overridden, so it is inherited
        assert haiku.concurrency == base.concurrency


class TestBucketMath:
    def test_a_full_bucket_admits_its_capacity_then_meters_the_refill(self):
        clock = FakeClock()
        lim = limiter(clock, requests=60)
        for _ in range(60):
            assert lim.acquire("openai", "gpt-5", "k", "requests", 1) == 0.0
        # 60/minute = one per second: the 61st waits exactly a second.
        assert lim.acquire("openai", "gpt-5", "k", "requests", 1) == pytest.approx(1.0)
        assert clock.slept == [1.0]

    def test_a_request_larger_than_the_whole_bucket_waits_for_a_full_one(self):
        clock = FakeClock()
        lim = limiter(clock, input_tokens=1000)
        lim.acquire("openai", "gpt-5", "k", "input_tokens", 1000)
        waited = lim.acquire("openai", "gpt-5", "k", "input_tokens", 5000)
        assert waited == pytest.approx(60.0)   # a full bucket, then go

    def test_an_unknown_currency_is_not_a_limit(self):
        clock = FakeClock()
        lim = limiter(clock, requests=10)
        assert lim.acquire("openai", "gpt-5", "k", "output_tokens", 10_000) == 0.0

    def test_released_tokens_come_back(self):
        clock = FakeClock()
        lim = limiter(clock, output_tokens=1000)
        lim.acquire("openai", "gpt-5", "k", "output_tokens", 1000)
        lim.release("openai", "gpt-5", "k", "output_tokens", 900)
        assert lim.acquire("openai", "gpt-5", "k", "output_tokens", 900) == 0.0


class TestLearningFromHeaders:
    def test_a_remaining_header_corrects_a_seeded_limit_downward(self):
        clock = FakeClock()
        lim = limiter(clock, requests=600)
        lim.observe("openai", "gpt-5", "k",
                    RateLimits(requests_remaining=2, requests_reset=30.0))
        for _ in range(2):
            assert lim.acquire("openai", "gpt-5", "k", "requests", 1) == 0.0
        # The seed said 600/minute; the account says two were left.
        assert lim.acquire("openai", "gpt-5", "k", "requests", 1) > 0.0

    def test_an_exhausted_currency_holds_the_scope_until_its_reset(self):
        clock = FakeClock()
        lim = limiter(clock, requests=600)
        lim.observe("anthropic", "claude-opus-5", "k",
                    RateLimits(requests_remaining=0, requests_reset=12.0))
        assert lim.acquire("anthropic", "claude-opus-5", "k", "requests", 1) == pytest.approx(12.0)

    def test_a_header_cannot_credit_more_than_the_capacity(self):
        clock = FakeClock()
        lim = limiter(clock, requests=10)
        lim.acquire("openai", "gpt-5", "k", "requests", 10)
        lim.observe("openai", "gpt-5", "k", RateLimits(requests_remaining=9999))
        # The provider's number wins over our local simulation, but it
        # is clamped to what we believe the window holds: ten, not 9999.
        assert lim.acquire("openai", "gpt-5", "k", "requests", 10) == 0.0
        assert lim.acquire("openai", "gpt-5", "k", "requests", 1) > 0.0

    def test_a_429_stops_the_scope_for_the_headers_reset_exactly(self):
        clock = FakeClock()
        lim = limiter(clock, requests=600)
        lim.penalize("xai", "grok-4", "k", 7.5)
        assert lim.acquire("xai", "grok-4", "k", "requests", 1) == pytest.approx(7.5)
        assert clock.slept == [7.5]

    def test_a_429_without_a_reset_still_pauses(self):
        clock = FakeClock()
        lim = limiter(clock, requests=600)
        lim.penalize("xai", "grok-4", "k", 0.0)
        assert lim.acquire("xai", "grok-4", "k", "requests", 1) == pytest.approx(1.0)


class TestThroughTheTransport:
    def test_the_transport_spends_every_currency_and_gives_back_the_unused(self):
        clock = FakeClock()
        lim = limiter(clock, requests=600, input_tokens=100_000,
                      output_tokens=1000, concurrency=4)
        t = MockTransport(name="openai", sleep=clock.sleep, clock=clock.now)
        out = t.chat(request({"model": "gpt-5", "max_tokens": 900,
                              "messages": [{"role": "user", "content": "hi"}]}),
                     limiter=lim, scope="k")
        wrote = out.usage.output_tokens
        assert 0 < wrote < 900
        # 900 reserved, `wrote` spent, the rest returned — so a second
        # call of the same shape still fits.
        assert lim.acquire("openai", "gpt-5", "k", "output_tokens", 900 - wrote) == 0.0
        # And the slot it held is back.
        assert lim.acquire("openai", "gpt-5", "k", "concurrency", 4) == 0.0

    def test_a_429_from_the_provider_reaches_the_limiter_as_a_hold(self):
        clock = FakeClock()
        lim = limiter(clock, requests=600)
        t = MockTransport(name="openai", script=[rate_limited(4.0)],
                          sleep=clock.sleep, clock=clock.now)
        t.chat(request({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}),
               limiter=lim, scope="k")
        # The retry slept the header's 4s; the scope is held for it too.
        assert 4.0 in clock.slept

    def test_the_concurrency_cap_holds_under_real_parallel_calls(self):
        lim = reg.TokenBucketLimiter(limits=lambda _p, _m="": reg.Limits(concurrency=3))
        peak = 0
        live = 0
        seen = threading.Lock()
        done = threading.Event()

        def one():
            nonlocal peak, live
            lim.acquire("openai", "gpt-5", "k", "concurrency", 1)
            with seen:
                live += 1
                peak = max(peak, live)
            done.wait(0.02)
            with seen:
                live -= 1
            lim.release("openai", "gpt-5", "k", "concurrency", 1)

        threads = [threading.Thread(target=one) for _ in range(12)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=10)
        assert peak <= 3
        assert all(not th.is_alive() for th in threads)


class TestFairnessAndSelfCorrection:
    """The two bugs that killed experiment 024's first ceiling run at
    154 of 200 calls: contending threads could starve each other, and a
    conservative seed could never be corrected upward."""

    def test_no_thread_is_overtaken_forever(self):
        # Eight threads, a bucket far too small for them: with no
        # queueing one loses every race and dies. With tickets, all of
        # them get through, in order.
        lim = reg.TokenBucketLimiter(
            limits=lambda _p, _m="": reg.Limits(output_tokens=6000),
            max_waits=64)
        done: list[int] = []
        errors: list[BaseException] = []

        def one(i):
            try:
                lim.acquire("anthropic", "claude-sonnet-5", "k",
                            "output_tokens", 250)
                done.append(i)
            except BaseException as e:  # noqa: BLE001 — the test is the assert
                errors.append(e)

        threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert errors == []
        assert sorted(done) == list(range(8))
        assert done == sorted(done)          # served in the order asked

    def test_a_header_can_raise_a_seed_that_was_too_low(self):
        clock = FakeClock()
        lim = limiter(clock, output_tokens=8000)
        lim.acquire("anthropic", "claude-sonnet-5", "k", "output_tokens", 8000)
        # The account is ten times what the registry guessed.
        lim.observe("anthropic", "claude-sonnet-5", "k",
                    RateLimits(output_tokens_limit=80_000,
                               output_tokens_remaining=64_000))
        assert lim.acquire("anthropic", "claude-sonnet-5", "k",
                           "output_tokens", 20_000) == 0.0

    def test_a_429_hold_outranks_any_remaining_header(self):
        # The provider's numbers win over our simulation — but a 429
        # outranks both, or a stale "plenty remaining" would walk us
        # straight back into the wall.
        clock = FakeClock()
        lim = limiter(clock, requests=600)
        lim.penalize("openai", "gpt-5", "k", 30.0)
        lim.observe("openai", "gpt-5", "k", RateLimits(requests_remaining=599))
        assert lim.acquire("openai", "gpt-5", "k", "requests", 1) == pytest.approx(30.0)

    def test_giving_up_is_an_interrupt_not_a_crash(self):
        # A job that cannot get quota should come back later with its
        # partials, not fail with a RuntimeError at 77%.
        from mechbench_compute.providers.errors import ProviderUnavailable

        clock = FakeClock()
        lim = reg.TokenBucketLimiter(
            sleep=clock.sleep, clock=lambda: 0.0,     # time never advances
            limits=lambda _p, _m="": reg.Limits(requests=60), max_waits=3)
        lim.acquire("anthropic", "m", "k", "requests", 60)
        with pytest.raises(ProviderUnavailable, match="did not free up"):
            lim.acquire("anthropic", "m", "k", "requests", 60)

    def test_the_anthropic_limit_headers_are_read(self):
        limits = RateLimits.from_headers({
            "anthropic-ratelimit-output-tokens-limit": "80000",
            "anthropic-ratelimit-output-tokens-remaining": "64000",
            "anthropic-ratelimit-input-tokens-limit": "200000",
        })
        assert limits.output_tokens_limit == 80_000
        assert limits.output_tokens_remaining == 64_000
        assert limits.input_tokens_limit == 200_000
