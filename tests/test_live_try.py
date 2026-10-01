from __future__ import annotations

import dataclasses

import pytest

from mechbench_compute import architectures
from mechbench_compute import model as model_mod
from mechbench_compute.api import TryRefused, run_try
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.live.run_try import TOO_LONG, summary_line
from mechbench_compute.protocol import ProtocolExecutor
from mechbench_compute.resume import content_hash
from tests.tiny_models import build_tiny_model

MODEL = "tiny/llama"
RECORDS = [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"}]


@pytest.fixture
def tiny_hub(monkeypatch, tmp_path):
    tiny = build_tiny_model("llama")
    loads = []
    architecture = dataclasses.replace(
        architectures.BY_MODEL_TYPE["llama"],
        load=lambda model_id, **_: loads.append(model_id) or (tiny._model, tiny._processor))
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": "llama"})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)
    return loads


def _warm(loads):
    ex = ProtocolExecutor()
    ex._model_loaded(MODEL)
    assert len(loads) == 1
    return ex


def test_a_try_reads_layers_on_the_warm_model_and_says_what_it_found(tiny_hub):
    ex = _warm(tiny_hub)
    got = run_try(ex, op="logits/read-layers", inputs={"records": RECORDS}, params={"top_k": 3},
                  model=MODEL, live_run_id="live_1", seq=4)
    assert len(tiny_hub) == 1
    assert got["kind"] == "logits/funnel"
    assert got["hash"] == content_hash(got["result"])
    assert len(K.items_of(got["result"])) == got["summary"]["items"] > 0
    assert got["lines"] == [summary_line(got["summary"])]
    assert got["provenance"]["op"] == "logits/read-layers"
    assert got["provenance"]["model"].startswith(MODEL)
    again = run_try(ex, op="logits/read-layers", inputs={"records": RECORDS}, params={"top_k": 3},
                    model=MODEL, live_run_id="live_1", seq=4)
    assert again["hash"] == got["hash"] and again["provenance"]["seed"] == got["provenance"]["seed"]
    other = run_try(ex, op="logits/read-layers", inputs={"records": RECORDS}, params={"top_k": 3},
                    model=MODEL, live_run_id="live_1", seq=5)
    assert other["provenance"]["seed"] != got["provenance"]["seed"]


def test_a_try_streams_tokens_and_stops_when_its_time_is_up(tiny_hub):
    ex = _warm(tiny_hub)
    seen = []
    got = run_try(ex, op="text/generate", inputs={"records": RECORDS[:1]},
                  params={"n": 1, "max_tokens": 3, "temperature": 0.0}, model=MODEL,
                  on_token=lambda *a: seen.append(a))
    assert got["kind"] is not None and seen
    ticks = iter([0.0, 0.0, 0.0, 0.0] + [999.0] * 100)
    with pytest.raises(TryRefused, match=TOO_LONG):
        run_try(ex, op="text/generate", inputs={"records": RECORDS[:1]},
                params={"n": 1, "max_tokens": 3, "temperature": 0.0}, model=MODEL,
                wall_seconds=1.0, clock=lambda: next(ticks))


def test_a_graph_runs_through_the_same_checks_and_answers_from_its_end():
    graph = {"nodes": [{"id": "u", "block": "records/union", "params": {}, "inputs": {}}],
             "edges": [{"from": {"input": "a"}, "to": {"node": "u", "port": "before"}},
                       {"from": {"input": "b"}, "to": {"node": "u", "port": "now"}}]}
    got = run_try(ProtocolExecutor(), graph=graph,
                  inputs={"a": [{"id": "1", "text": "x"}], "b": [{"id": "2", "text": "y"}]},
                  params={}, model=MODEL)
    assert sorted(i["id"] for i in K.items_of(got["result"])) == ["1", "2"]
    assert got["provenance"]["op"] == ["records/union"]
    unbound = {"nodes": [{"id": "s", "block": "records/sort", "params": {"by": {"$param": "nope"}},
                          "inputs": {"records": []}}], "edges": []}
    with pytest.raises(ValueError, match="unbound param"):
        run_try(ProtocolExecutor(), graph=unbound, inputs={}, params={}, model=MODEL)


@pytest.mark.parametrize(("op", "params", "said"), [
    ("adapter/train", {}, "a job, not a try"),
    ("text/chat", {"model": {"provider": "openai", "model": "gpt-4o"}}, "hosted"),
    ("dictionary/load", {}, "reaches the network"),
    ("logits/read", {"model": "google/gemma-3-12b-it"}, f"this live run holds {MODEL}"),
    ("nobody/nothing", {}, "run it as a job"),
])
def test_a_try_refuses_what_belongs_in_a_job(op, params, said):
    with pytest.raises(TryRefused, match=said):
        run_try(ProtocolExecutor(), op=op, inputs={}, params=params, model=MODEL)


def test_a_try_refuses_a_graph_over_sixteen_nodes_or_with_two_ends():
    many = {"nodes": [{"id": f"n{i}", "block": "records/union", "params": {}, "inputs": {}}
                      for i in range(17)], "edges": []}
    with pytest.raises(TryRefused, match="at most 16"):
        run_try(ProtocolExecutor(), graph=many, inputs={}, params={}, model=MODEL)
    two = {**many, "nodes": many["nodes"][:2]}
    with pytest.raises(TryRefused, match="ends in one node"):
        run_try(ProtocolExecutor(), graph=two, inputs={}, params={}, model=MODEL)


def test_a_kind_that_speaks_says_its_first_five_items():
    from mechbench_compute.live.run_try import speak_lines

    items = [{"id": f"g{i}", "a_layer": i, "b_layer": i + 1, "score": 0.5} for i in range(7)]
    value = K.collection("geometry/alignment", items, header={"method": "cka"})
    lines = speak_lines(value, {"kind": "geometry/alignment", "collection": True, "items": 7})
    assert lines[0] == "cka alignment of layers 0 and 1: 0.5" and len(lines) == 5
