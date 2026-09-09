"""The provider transport: canonical requests, metering, limits,
adapters, and the two ways a test never spends (tasks 000337 / 000350).

Nothing here touches the network. The adapters are exercised against a
captured `post_json`, which is where the wire mapping actually lives.
"""

from __future__ import annotations

import pytest

from mechbench_compute.providers import (
    Budget,
    Cassette,
    CassetteTransport,
    budget_from,
    capabilities,
    make_transport,
    pricing,
)
from mechbench_compute.providers import (
    messages as m,
)
from mechbench_compute.providers.base import Capabilities
from mechbench_compute.providers.errors import (
    BudgetExceeded,
    CapabilityUnsupported,
    CassetteMiss,
    ProviderUnavailable,
)
from mechbench_compute.providers.limiter import RateLimits, RecordingLimiter
from mechbench_compute.providers.mock import MockTransport, rate_limited, transient


def req(**kw):
    base = {"model": "claude-opus-5", "system": "be brief",
            "messages": [{"role": "user", "content": "hello"}], "max_tokens": 64}
    base.update(kw)
    return m.request(base)


class TestCanonicalRequest:
    def test_the_hash_covers_what_was_asked_and_nothing_about_auth(self):
        c = m.canonical(req())
        assert set(c) == {"model", "messages", "max_tokens", "system"}
        assert m.request_hash(req()) == m.request_hash(req())
        assert m.request_hash(req(max_tokens=65)) != m.request_hash(req())

    def test_absent_optional_fields_are_omitted_not_null(self):
        # So adding a field later cannot change the hash of a request
        # that never set it.
        assert "temperature" not in m.canonical(req())
        assert m.canonical(req(temperature=0.0))["temperature"] == 0.0

    def test_provider_options_are_scoped_and_hashed_per_provider(self):
        r = req(provider_options={"anthropic": {"thinking": {"budget_tokens": 1024}},
                                  "openai": {"service_tier": "flex"}})
        anth = m.canonical(r, provider="anthropic")
        assert anth["provider_options"] == {
            "anthropic": {"thinking": {"budget_tokens": 1024}}}
        assert m.request_hash(r, provider="anthropic") != m.request_hash(r, provider="openai")
        assert r.options_for("xai") == {}

    def test_a_system_turn_in_the_message_list_is_refused(self):
        with pytest.raises(ValueError, match="field of the request"):
            m.messages([{"role": "system", "content": "no"}])

    def test_an_unknown_request_field_is_refused_by_name(self):
        with pytest.raises(ValueError, match="max_token"):
            m.request({"model": "x", "max_token": 10})

    def test_tool_parts_round_trip(self):
        conv = m.messages([
            {"role": "assistant", "content": [
                {"type": "tool_call", "id": "c1", "name": "grep",
                 "arguments": {"pattern": "x"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_call_id": "c1", "content": "3 hits"}]},
        ])
        wire = [x.to_wire() for x in conv]
        assert m.messages(wire) == conv


class TestMockDeterminism:
    def test_the_same_request_always_gets_the_same_answer(self):
        a, b = MockTransport(), MockTransport()
        assert a.chat(req()).text == b.chat(req()).text

    def test_a_different_request_gets_a_different_answer(self):
        t = MockTransport()
        assert t.chat(req()).text != t.chat(req(system="be long")).text

    def test_a_tool_call_comes_back_as_a_part(self):
        t = MockTransport()
        r = req(tools=[{"name": "grep", "input_schema": {
            "type": "object", "properties": {"pattern": {"type": "string"}}}}],
            provider_options={"mock": {"tool_call": "grep"}})
        out = t.chat(r)
        assert out.stop_reason == "tool_use"
        assert out.tool_calls[0].name == "grep"
        assert out.as_message().role == "assistant"


class TestBudget:
    def test_a_call_that_could_cross_the_cap_is_refused_before_it_is_made(self):
        t = MockTransport(name="anthropic", capabilities=capabilities("anthropic"))
        # max_tokens 4000 of opus output is $0.30 — over a $0.10 cap.
        b = Budget(cap_usd=0.10)
        with pytest.raises(BudgetExceeded, match=r"worst case"):
            t.chat(req(max_tokens=4000), budget=b)
        assert t.calls == []            # nothing left the machine
        assert b.spent_usd == 0.0 and b.reserved_usd == 0.0

    def test_spend_settles_to_the_real_usage(self):
        t = MockTransport(name="anthropic", capabilities=capabilities("anthropic"))
        b = Budget(cap_usd=1.0)
        out = t.chat(req(max_tokens=200), budget=b)
        assert b.spent_usd == pytest.approx(out.call.cost_usd)
        assert 0 < b.spent_usd < 0.02 and b.reserved_usd == 0.0 and b.calls == 1

    def test_a_node_budget_is_bounded_by_its_job_budget(self):
        job = Budget(cap_usd=0.05)
        node = job.child(cap_usd=10.0)
        assert node.cap_usd == 0.05
        t = MockTransport(name="anthropic", capabilities=capabilities("anthropic"))
        with pytest.raises(BudgetExceeded):
            t.chat(req(max_tokens=4000), budget=node)
        assert job.reserved_usd == 0.0

    def test_a_remote_node_without_a_budget_is_refused(self):
        with pytest.raises(ValueError, match="budget_usd"):
            budget_from({"model": {"provider": "anthropic"}})
        assert budget_from({"budget_usd": 2}).cap_usd == 2.0

    def test_unknown_models_are_flagged_not_guessed(self):
        cost, priced = pricing.cost_usd("anthropic", "claude-from-2031",
                                        {"input_tokens": 1000, "output_tokens": 1000})
        assert (cost, priced) == (0.0, False)

    def test_cached_input_is_priced_at_its_own_rate(self):
        cost, _ = pricing.cost_usd("anthropic", "claude-opus-5", {
            "input_tokens": 1_000_000, "output_tokens": 0,
            "cache_read_tokens": 900_000})
        assert cost == pytest.approx(100_000 / 1e6 * 15.0 + 900_000 / 1e6 * 1.5)


class TestCapabilities:
    def test_asking_a_provider_for_what_it_lacks_is_refused_by_name(self):
        t = MockTransport(name="anthropic", capabilities=capabilities("anthropic"))
        with pytest.raises(CapabilityUnsupported, match="logprobs"):
            t.chat(req(logprobs=5))
        with pytest.raises(CapabilityUnsupported, match="seed"):
            t.chat(req(seed=7))

    def test_a_top_k_over_the_providers_limit_names_the_limit(self):
        t = MockTransport(name="openai", capabilities=capabilities("openai"))
        with pytest.raises(CapabilityUnsupported, match="20 is the limit"):
            t.chat(req(logprobs=50))
        assert t.chat(req(logprobs=5)).logprobs is not None


class TestRetriesAndLimits:
    def test_a_429_waits_the_headers_reset_and_the_limiter_hears_about_it(self):
        slept: list[float] = []
        t = MockTransport(script=[rate_limited(2.5), transient(), None],
                          sleep=slept.append)
        lim = RecordingLimiter()
        out = t.chat(req(), limiter=lim)
        assert slept == [2.5, 1.0]                 # header's reset, then backoff
        assert out.call.attempts == 3
        assert out.call.throttled_seconds == pytest.approx(3.5)
        assert lim.penalties[0][3] == 2.5
        assert [c[3] for c in lim.acquired] == [
            "concurrency", "requests", "input_tokens", "output_tokens"]

    def test_a_sustained_failure_window_becomes_an_interruptible_outage(self):
        clock = iter([0.0, 0.0, 100.0, 700.0, 700.0, 700.0])
        t = MockTransport(script=[transient()] * 5, sleep=lambda _s: None,
                          clock=lambda: next(clock))
        t.outage_seconds = 600.0
        with pytest.raises(ProviderUnavailable, match="interrupting"):
            t.chat(req())

    def test_rate_limit_headers_are_read_from_either_dialect(self):
        a = RateLimits.from_headers({"anthropic-ratelimit-requests-remaining": "3",
                                     "anthropic-ratelimit-tokens-reset": "1.5s"})
        assert (a.requests_remaining, a.tokens_reset) == (3, 1.5)
        o = RateLimits.from_headers({"x-ratelimit-remaining-requests": "9",
                                     "x-ratelimit-reset-requests": "350ms",
                                     "retry-after": "20"})
        assert (o.requests_remaining, o.retry_after) == (9, 20.0)
        assert o.requests_reset == pytest.approx(0.35)
        assert RateLimits.from_headers({"retry-after": "not-a-time"}).retry_after is None


class TestCassettes:
    def test_record_then_replay_is_the_same_answer_with_no_inner_adapter(self):
        inner = MockTransport()
        tape = Cassette(provider="mock", label="fixtures/hello")
        rec = CassetteTransport(tape, inner=inner, mode="record")
        first = rec.chat(req()).text

        replay = CassetteTransport(Cassette.from_wire(tape.to_wire()), mode="replay")
        assert replay.chat(req()).text == first
        assert replay.chat(req()).call.provider == "mock"

    def test_a_request_that_drifted_misses_loudly(self):
        tape = Cassette(provider="mock")
        CassetteTransport(tape, inner=MockTransport(), mode="record").chat(req())
        replay = CassetteTransport(tape, mode="replay")
        with pytest.raises(CassetteMiss, match="no recorded response"):
            replay.chat(req(system="a different system prompt"))

    def test_repeated_identical_requests_replay_in_order(self):
        tape = Cassette(provider="mock")
        rec = CassetteTransport(tape, inner=MockTransport(), mode="record")
        rec.cassette.add(m.request_hash(req(), provider="mock"),
                         _canned("first"))
        rec.cassette.add(m.request_hash(req(), provider="mock"),
                         _canned("second"))
        replay = CassetteTransport(tape, mode="replay")
        assert [replay.chat(req()).text for _ in range(3)] == [
            "first", "second", "second"]

    def test_secrets_and_auth_headers_never_enter_a_cassette(self):
        inner = MockTransport(headers={"authorization": "Bearer sk-live-123",
                                       "anthropic-ratelimit-requests-remaining": "7",
                                       "x-account": "sk-live-123"})
        tape = Cassette(provider="mock")
        CassetteTransport(tape, inner=inner, mode="record",
                          secrets=("sk-live-123",)).chat(req())
        stored = str(tape.to_wire())
        assert "sk-live-123" not in stored and "authorization" not in stored
        assert "anthropic-ratelimit-requests-remaining" in stored

    def test_a_dry_run_impersonates_the_provider_it_stands_in_for(self):
        t = make_transport("anthropic", dry_run=True)
        assert t.name == "anthropic"
        with pytest.raises(CapabilityUnsupported):
            t.chat(req(logprobs=5))          # the real refusal, for free
        assert t.chat(req()).call.cost_usd > 0    # the real price table


def _canned(text: str):
    from mechbench_compute.providers.base import AdapterResponse, Usage

    return AdapterResponse(parts=(m.TextPart(text),), usage=Usage(10, 5),
                           model_version="canned", response_id="r1")


# --- the wire mappings, without a network ---------------------------------------


class _Capture:
    """Stands in for providers.http.post_json."""

    def __init__(self, body):
        self.body = body
        self.sent: dict = {}

    def __call__(self, url, *, headers, payload, timeout, secrets=()):
        from mechbench_compute.providers.http import HttpResponse

        self.sent = {"url": url, "headers": headers, "payload": payload}
        return HttpResponse(status=200, headers={"x-request-id": "req_1"},
                            body=self.body)


TOOLS = [{"name": "grep", "description": "search",
          "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}]

CONVERSATION = [
    {"role": "user", "content": "find it"},
    {"role": "assistant", "content": [
        {"type": "tool_call", "id": "c1", "name": "grep", "arguments": {"q": "x"}}]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_call_id": "c1", "content": "3 hits"},
        {"type": "text", "text": "what now?"}]},
]


class TestAnthropicMapping:
    def _run(self, monkeypatch, body):
        from mechbench_compute.providers import anthropic, http

        cap = _Capture(body)
        monkeypatch.setattr(http, "post_json", cap)
        t = anthropic.AnthropicTransport({"token": "sk-ant-x"})
        out = t.chat(m.request({
            "model": "claude-opus-5", "system": "be brief",
            "messages": CONVERSATION, "tools": TOOLS, "max_tokens": 100,
            "provider_options": {"anthropic": {"thinking": {"type": "enabled"}}}}))
        return cap, out

    def test_the_body_is_the_messages_api_and_options_ride_verbatim(self, monkeypatch):
        cap, out = self._run(monkeypatch, {
            "id": "msg_1", "model": "claude-opus-5-20260101",
            "content": [{"type": "text", "text": "found"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 90, "output_tokens": 12,
                      "cache_read_input_tokens": 40}})
        p = cap.sent["payload"]
        assert cap.sent["headers"]["x-api-key"] == "sk-ant-x"
        assert p["system"] == "be brief" and p["tools"][0]["name"] == "grep"
        assert p["messages"][1]["content"][0] == {
            "type": "tool_use", "id": "c1", "name": "grep", "input": {"q": "x"}}
        assert p["messages"][2]["content"][0]["type"] == "tool_result"
        assert p["thinking"] == {"type": "enabled"}
        # Provenance: who answered, what it used, what it cost.
        assert out.call.model_version == "claude-opus-5-20260101"
        assert out.call.usage["cache_read_tokens"] == 40
        assert out.call.provider_options == {"thinking": {"type": "enabled"}}
        assert out.call.cost_usd > 0 and out.text == "found"

    def test_a_tool_use_response_becomes_a_tool_call_part(self, monkeypatch):
        _, out = self._run(monkeypatch, {
            "id": "msg_2", "model": "claude-opus-5",
            "content": [{"type": "tool_use", "id": "t1", "name": "grep",
                         "input": {"q": "y"}}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 5}})
        assert out.tool_calls[0].arguments == {"q": "y"}


class TestOpenAIMapping:
    def test_tool_calls_split_into_their_own_messages(self, monkeypatch):
        from mechbench_compute.providers import http, openai_compatible

        cap = _Capture({"id": "cc1", "model": "gpt-5-2026",
                        "choices": [{"message": {"content": "ok"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 4,
                                  "prompt_tokens_details": {"cached_tokens": 20}}})
        monkeypatch.setattr(http, "post_json", cap)
        t = openai_compatible.OpenAICompatibleTransport({"token": "sk-x"},
                                                        provider="openai")
        out = t.chat(m.request({"model": "gpt-5", "system": "be brief",
                                "messages": CONVERSATION, "tools": TOOLS,
                                "seed": 3, "json_mode": True, "max_tokens": 50}))
        roles = [x["role"] for x in cap.sent["payload"]["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "user"]
        call = cap.sent["payload"]["messages"][2]["tool_calls"][0]
        assert call["function"] == {"name": "grep", "arguments": '{"q": "x"}'}
        assert cap.sent["payload"]["response_format"] == {"type": "json_object"}
        assert cap.sent["payload"]["seed"] == 3
        assert out.call.usage["cache_read_tokens"] == 20

    def test_a_local_server_needs_no_key_but_does_need_a_base_url(self):
        from mechbench_compute.providers import openai_compatible as oc

        t = oc.OpenAICompatibleTransport({"base_url": "http://127.0.0.1:8080/v1"},
                                         provider="openai-compatible")
        assert t._headers() == {}
        with pytest.raises(Exception, match="base_url"):
            oc.OpenAICompatibleTransport({}, provider="openai-compatible")


class TestGeminiMapping:
    def test_contents_roles_and_function_responses(self, monkeypatch):
        from mechbench_compute.providers import gemini, http

        cap = _Capture({"modelVersion": "gemini-2.5-pro-002",
                        "responseId": "g1",
                        "candidates": [{"content": {"parts": [{"text": "found"}]},
                                        "finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 60,
                                          "candidatesTokenCount": 8}})
        monkeypatch.setattr(http, "post_json", cap)
        t = gemini.GeminiTransport("key-1")
        out = t.chat(m.request({"model": "gemini-2.5-pro", "system": "be brief",
                                "messages": CONVERSATION, "tools": TOOLS,
                                "max_tokens": 100}))
        p = cap.sent["payload"]
        assert cap.sent["headers"] == {"x-goog-api-key": "key-1"}
        assert "key-1" not in cap.sent["url"]        # never in a URL
        assert [c["role"] for c in p["contents"]] == ["user", "model", "user"]
        # A tool result is keyed by the function's NAME here, resolved
        # from the call it answers.
        assert p["contents"][2]["parts"][0]["functionResponse"]["name"] == "grep"
        assert p["systemInstruction"]["parts"][0]["text"] == "be brief"
        assert p["tools"][0]["functionDeclarations"][0]["name"] == "grep"
        assert out.call.model_version == "gemini-2.5-pro-002"


class TestFactory:
    def test_every_provider_declares_a_capability_matrix(self):
        from mechbench_compute.providers import PROVIDERS

        for p in PROVIDERS:
            assert isinstance(capabilities(p), Capabilities)
        with pytest.raises(ValueError, match="unknown provider"):
            capabilities("altavista")

    def test_replay_needs_no_credential_at_all(self):
        tape = Cassette(provider="anthropic")
        t = make_transport("anthropic", None, cassette=tape, cassette_mode="replay")
        assert isinstance(t, CassetteTransport) and t.inner is None
