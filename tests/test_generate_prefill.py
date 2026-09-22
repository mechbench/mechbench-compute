"""`text/generate` continuing a record's prefill and stopping at a
marker: the samples continue exactly the envelope a decision read and a
training step condition on, and end where the answer does — on a fake
substrate, so what reaches the sampler is the test's to inspect."""

from __future__ import annotations

import pytest

from mechbench_compute import distill, generate
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec


class _Tok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _Model:
    tokenizer = _Tok()


RECORD = {"id": "genre-p0", "system": "Pick a genre.", "user": "One, please.",
          "prefill": '{ "genre": "'}


@pytest.fixture
def seen(monkeypatch):
    """The fake sampler answers from a script keyed by sample index and
    records the prompt it was given and the stop strings it was asked to
    honour."""
    calls: list[dict] = []
    script = iter(['Steampunk"', "Science Fiction and more", "Hum"])

    monkeypatch.setattr(ProtocolExecutor, "_model_loaded", lambda self, model_id: _Model())
    monkeypatch.setattr(ProtocolExecutor, "_run_model_block",
                        lambda self, fn, inputs, params, *a, **k: fn(inputs, params, *a, **k))
    monkeypatch.setattr(distill, "render_chat", lambda tok, s, u, p: f"<{s}|{u}>{p}")
    monkeypatch.setattr(distill, "encode", lambda tok, text: [ord(c) for c in text])
    monkeypatch.setattr(distill, "prefill_decision", lambda model, ids: ("cache", ids))

    def sample(model, ids, *, max_tokens, temperature, top_p, rng, prefill,
               return_ids=False, stop_strings=()):
        raw = next(script)
        calls.append({"prompt": "".join(chr(i) for i in ids), "stop": stop_strings})
        cut = raw
        for s in stop_strings:
            if s in cut:
                cut = cut[:cut.index(s)]
        return cut, [ord(c) for c in raw[:max_tokens]]

    monkeypatch.setattr(generate, "sample_completion_cached", sample)
    return calls


def _run(params):
    graph = {"nodes": [{"id": "gen", "block": "text/generate",
                        "params": {"model": "fake/m@rev", "n": 3, "seed": 7, **params},
                        "inputs": {"records": [RECORD]}}], "edges": []}
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                              extra={"graph": graph}))
    payload = out.payload if hasattr(out, "payload") else out
    return payload["outputs"]["gen"]["items"]


def test_the_prefill_begins_the_turn_and_the_answer_stops_at_its_marker(seen):
    items = _run({"continue_prefill": True, "stop": ['"'], "max_tokens": 20})
    assert all(c["prompt"].endswith('{ "genre": "') for c in seen)
    assert all(c["stop"] == ('"',) for c in seen)
    assert items[0]["text"] == '{ "genre": "Steampunk'
    assert items[0]["metadata"]["sampling"]["ended"] == "stop"
    assert items[0]["metadata"]["sampling"]["prefill"] == '{ "genre": "'
    # Never closed: it ran out of tokens, and says so.
    assert items[1]["metadata"]["sampling"]["ended"] == "max_tokens"
    # Ended its turn before the marker.
    assert items[2]["text"] == '{ "genre": "Hum'
    assert items[2]["metadata"]["sampling"]["ended"] == "end"


def test_without_it_the_prefill_is_dropped_as_before(seen):
    items = _run({"max_tokens": 40})
    assert not any(c["prompt"].endswith('{ "genre": "') for c in seen)
    assert items[0]["text"] == 'Steampunk"'
    sampling = items[0]["metadata"]["sampling"]
    assert "prefill" not in sampling and "stop" not in sampling and sampling["ended"] == "end"
