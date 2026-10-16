from __future__ import annotations

import copy
import re

import pytest

from mechbench_compute import chat as chat_mod
from mechbench_compute import model_ref as mr
from mechbench_compute.ops.eval.judge import run_judge
from mechbench_compute.providers import capabilities, endpoint, http
from mechbench_compute.providers import messages as m
from mechbench_compute.providers.errors import ProviderError
from mechbench_compute.providers.features import FEATURES
from mechbench_compute.providers.mock import MockTransport
from mechbench_compute.providers.provider_client import ProviderClient

SAMPLING = ("temperature", "top_p", "top_k")


class Script:
    def __init__(self, text: str = "Four.", *, refuse: str | None = None):
        self.text = text
        self.refuse = refuse
        self.sent: list[dict] = []

    def __call__(self, url, *, headers, payload, timeout, secrets=()):
        if url.endswith("count_tokens"):
            return http.HttpResponse(status=200, headers={}, body={"input_tokens": 10})
        self.sent.append(copy.deepcopy(payload))
        if self.refuse:
            raise ProviderError(f"HTTP 400: {self.refuse!r}")
        return http.HttpResponse(status=200, headers={}, body={
            "id": "m1", "model": payload["model"], "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3},
            "content": [{"type": "text", "text": self.text}]})


@pytest.fixture
def wire(monkeypatch):
    def install(script: Script) -> Script:
        monkeypatch.setattr(http, "post_json", script)
        monkeypatch.setattr(endpoint, "resolve_addresses", lambda _host, _port: ["104.18.2.3"])
        return script
    return install


def chat(model: str, **params):
    ref = mr.parse({"provider": "anthropic", "model": model})
    return chat_mod.run_remote(
        ref, [{"id": "r0", "user": "What is 2+2?"}],
        {"budget_usd": 5.0, "concurrency": 1, **params},
        secrets={"anthropic": {"token": "k"}})


def judge(model: str, **judge_params):
    ref = mr.parse({"provider": "anthropic", "model": model})
    return run_judge(
        {"judge": {"model": {"provider": "anthropic", "model": model},
                   "system": "Grade the story.", **judge_params},
         "scale": {"type": "numeric", "min": 1, "max": 5}, "budget_usd": 1.0},
        inputs={"records": [{"id": "s1", "text": "The dust settled slowly."}]},
        provider=ProviderClient(ref, secrets={"anthropic": {"token": "k"}}))


class TestTextChat:
    def test_a_model_that_refuses_sampling_gets_the_request_without_it(self, wire):
        script = wire(Script())
        out = chat("claude-sonnet-5-5", temperature=0, top_p=0.9,
                   provider_options={"anthropic": {"top_k": 5}})
        assert len(script.sent) == 1
        assert not set(SAMPLING) & set(script.sent[0])
        why = "not accepted by claude-sonnet-5-5"
        assert out["omitted"] == {"temperature": why, "top_p": why, "top_k": why}
        item = out["items"][0]
        assert item["text"] == "Four."
        assert item["metadata"]["call"]["omitted"]["temperature"] == why
        assert item["metadata"]["sampling"]["temperature"] is None

    def test_a_model_that_takes_sampling_is_sent_it_and_nothing_is_noted(self, wire):
        script = wire(Script())
        out = chat("claude-haiku-4-5", temperature=0)
        assert script.sent[0]["temperature"] == 0
        assert "omitted" not in out
        item = out["items"][0]
        assert "omitted" not in item["metadata"]["call"]
        assert item["metadata"]["sampling"]["temperature"] == 0

    def test_parameters_left_unset_need_no_note(self, wire):
        script = wire(Script())
        out = chat("claude-sonnet-5-5")
        assert not set(SAMPLING) & set(script.sent[0])
        assert "omitted" not in out

    def test_a_refusal_of_any_other_kind_still_fails_the_node(self, wire):
        wire(Script(refuse="max_tokens: must be at most 64000"))
        with pytest.raises(ProviderError, match="max_tokens"):
            chat("claude-sonnet-5-5", temperature=0)


class TestJudge:
    def test_a_judge_that_refuses_temperature_judges_without_it(self, wire):
        script = wire(Script('{"score": 3, "rationale": "plain"}'))
        out = judge("claude-sonnet-5-5", temperature=0)
        assert "temperature" not in script.sent[0]
        assert out["omitted"] == {"temperature": "not accepted by claude-sonnet-5-5"}
        assert out["items"][0]["score"] == 3.0

    def test_a_judge_that_takes_temperature_is_sent_it(self, wire):
        script = wire(Script('{"score": 3, "rationale": "plain"}'))
        out = judge("claude-haiku-4-5", temperature=0)
        assert script.sent[0]["temperature"] == 0
        assert "omitted" not in out


class TestTransport:
    def test_a_dry_run_leaves_out_what_the_real_request_would(self):
        t = MockTransport(name="anthropic", capabilities=capabilities("anthropic"))
        out = t.chat(m.request({"model": "claude-opus-5-5", "temperature": 0.0,
                                "messages": [{"role": "user", "content": "hi"}]}))
        assert t.calls[0].temperature is None
        assert out.call.omitted == {"temperature": "not accepted by claude-opus-5-5"}

    def test_another_provider_is_sent_what_was_set(self):
        t = MockTransport(name="openai", capabilities=capabilities("openai"))
        out = t.chat(m.request({"model": "gpt-5.5", "temperature": 0.0,
                                "messages": [{"role": "user", "content": "hi"}]}))
        assert t.calls[0].temperature == 0.0
        assert out.call.omitted == {}


def claude_version(model: str) -> tuple[int, int]:
    found = re.fullmatch(r"claude-[a-z]+-(\d+)(?:-(\d+))?", model)
    assert found, f"{model} is not a Claude id this test can read a version from"
    return int(found.group(1)), int(found.group(2) or 0)


ANTHROPIC = sorted(FEATURES["anthropic"])


@pytest.mark.parametrize("model", ANTHROPIC)
def test_claude_4_7_and_later_take_no_sampling_parameters(model):
    assert FEATURES["anthropic"][model].sampling is (claude_version(model) < (4, 7))


@pytest.mark.parametrize("provider", sorted(set(FEATURES) - {"anthropic"}))
def test_every_other_provider_takes_them(provider):
    assert all(f.sampling for f in FEATURES[provider].values())
