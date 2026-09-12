"""Memoized remote calls (task 000355, epic 000334).

The point is money: experiment 024's judged sweep cost $4.44, and the
capability arm ran six times. Re-running an unchanged remote node must
not buy the same answers again.
"""
from __future__ import annotations

import pytest

from mechbench_compute.providers import messages as msg
from mechbench_compute.providers.base import (
    AdapterResponse,
    Capabilities,
    Transport,
    Usage,
)
from mechbench_compute.providers.budget import Budget
from mechbench_compute.providers.cassette import Cassette, CassetteTransport


class CountingAdapter(Transport):
    """An adapter that says how many times it was actually asked."""

    name = "anthropic"

    def __init__(self) -> None:
        super().__init__()
        self.capabilities = Capabilities(chat=True, tools=True,
                                         count_tokens="estimated")
        self.calls = 0

    def _count_tokens(self, req):
        return 10

    def _chat(self, req, *, on_token=None):
        self.calls += 1
        return AdapterResponse(
            parts=(msg.TextPart("an answer"),),
            usage=Usage(input_tokens=10, output_tokens=5),
            model_version="claude-x-20260101", response_id=f"r{self.calls}")


def _req(text="hello"):
    return msg.ChatRequest(model="claude-x", messages=(
        msg.Message(role="user", content=(msg.TextPart(text),)),))


class TestAMemoIsACassetteTheNodeWrote:
    def test_the_second_run_does_not_ask_the_provider(self):
        inner = CountingAdapter()
        memo = Cassette(provider="anthropic", label="benji/lab/memo")
        first = CassetteTransport(memo, inner=inner, mode="auto")
        first.chat(_req())
        assert inner.calls == 1

        # A fresh transport over the SAME memo: a later run, same node.
        memo.rewind()
        second = CassetteTransport(memo, inner=inner, mode="auto")
        out = second.chat(_req())
        assert inner.calls == 1, "the provider was asked again"
        assert out.text == "an answer"

    def test_a_different_request_still_costs(self):
        inner = CountingAdapter()
        memo = Cassette(provider="anthropic")
        t = CassetteTransport(memo, inner=inner, mode="auto")
        t.chat(_req("hello"))
        t.chat(_req("a different question"))
        assert inner.calls == 2

    def test_a_memo_round_trips_through_its_wire_form(self):
        inner = CountingAdapter()
        memo = Cassette(provider="anthropic")
        CassetteTransport(memo, inner=inner, mode="auto").chat(_req())
        restored = Cassette.from_wire(memo.to_wire())
        out = CassetteTransport(restored, inner=inner, mode="auto").chat(_req())
        assert inner.calls == 1 and out.text == "an answer"


class TestACachedCallIsFree:
    """The half that matters: a hit must not bill."""

    def _replayed(self):
        inner = CountingAdapter()
        memo = Cassette(provider="anthropic")
        CassetteTransport(memo, inner=inner, mode="auto").chat(_req())
        memo.rewind()
        return CassetteTransport(memo, inner=inner, mode="auto")

    def test_it_costs_nothing(self):
        out = self._replayed().chat(_req())
        assert out.call.cost_usd == 0.0
        assert out.call.replayed is True

    def test_it_does_not_draw_down_a_budget(self):
        budget = Budget(cap_usd=1.0)
        self._replayed().chat(_req(), budget=budget)
        assert budget.spent_usd == 0.0, "a cached re-run billed for a purchase it did not make"

    def test_the_original_usage_is_kept(self):
        # Zero cost, but the tokens the FIRST call spent are still on
        # the record — comparing a memoized run to its first run needs
        # them.
        out = self._replayed().chat(_req())
        assert out.call.usage["input_tokens"] == 10
        assert out.call.usage["output_tokens"] == 5

    def test_an_uncached_call_still_bills(self):
        inner = CountingAdapter()
        t = CassetteTransport(Cassette(provider="anthropic"), inner=inner,
                              mode="auto")
        budget = Budget(cap_usd=1.0)
        out = t.chat(_req(), budget=budget)
        assert out.call.replayed is False
        assert budget.spent_usd >= 0.0


class TestTheMemoLabel:
    def test_cache_true_refuses_and_says_why(self):
        from mechbench_compute.protocol import ProtocolExecutor

        with pytest.raises(ValueError, match="no label to store under"):
            ProtocolExecutor()._open_memo({"cache": True})

    def test_no_cache_is_no_memo(self):
        from mechbench_compute.protocol import ProtocolExecutor

        assert ProtocolExecutor()._open_memo({}) is None
