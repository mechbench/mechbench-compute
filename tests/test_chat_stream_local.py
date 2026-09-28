from __future__ import annotations

import json

import numpy as np

from mechbench_compute import chat as chat_mod
from mechbench_compute import model_ref as mr

DIRECTION = {"kind": "direction/vector", "vector": [1.0, 0.0],
             "space": {"model": None, "layer": 1, "point": "resid_post", "head": None, "d": 2}}


class _Tok:
    unk_token_id = 3

    def convert_tokens_to_ids(self, token):
        return 3

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, **kw):
        return repr(turns)


def _fake_sample(model, ids, *, on_token=None, readout=None, pieces_out=None, **kw):
    for i, (piece, x) in enumerate((("Hi", 0.5), (" there", -1.25))):
        coord = readout.read({readout.hook: np.array([[[x, 9.0]]], dtype=np.float32)}) if readout else None
        if pieces_out is not None:
            pieces_out.append(piece)
        if on_token is not None:
            on_token({"index": i, "id": 10 + i, "text": piece, **({"coord": coord} if coord is not None else {})})
    return "Hi there", [10, 11]


def _run(monkeypatch, inputs, stream):
    from mechbench_compute import distill, generate

    class FakeModel:
        tokenizer = _Tok()

    monkeypatch.setattr(distill, "encode", lambda t, text: [1, 2, 3])
    monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
    monkeypatch.setattr(generate, "sample_completion_cached", _fake_sample)
    return chat_mod.run_local(FakeModel(), mr.parse("google/gemma-4-e4b-it"),
                              [{"id": "r0", "user": "Hello?"}],
                              {"n": 1, "seed": 3, "_on_token": stream}, inputs=inputs)


def test_a_projected_reply_carries_its_tokens_and_coordinates_and_streams_them(monkeypatch):
    seen = []
    out = _run(monkeypatch, {"project": DIRECTION}, lambda key, e: seen.append((key, e)))
    item = out["items"][0]
    assert item["projection"]["tokens"] == ["Hi", " there"]
    assert item["projection"]["coords"] == [0.5, -1.25]
    assert item["projection"]["direction"]["space"]["layer"] == 1
    assert [(k, e["text"], e["coord"], e["round"]) for k, e in seen] == [
        ("r0:0", "Hi", 0.5, 0), ("r0:0", " there", -1.25, 0)]
    json.dumps(out)


def test_without_a_direction_the_reply_is_as_before_and_still_streams(monkeypatch):
    seen = []
    item = _run(monkeypatch, {}, lambda key, e: seen.append(e))["items"][0]
    assert "projection" not in item
    assert [e["text"] for e in seen] == ["Hi", " there"]
