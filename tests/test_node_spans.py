from __future__ import annotations

import dataclasses

import pytest

from mechbench_compute import architectures, model as model_mod
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.spans import SPAN_FIELDS, add_to_span, open_span
from tests.tiny_models import build_tiny_model

MODEL = "tiny/llama"


@pytest.fixture
def tiny_hub(monkeypatch, tmp_path):
    tiny = build_tiny_model("llama")
    architecture = dataclasses.replace(
        architectures.BY_MODEL_TYPE["llama"],
        load=lambda model_id, **_: (tiny._model, tiny._processor))
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": "llama"})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)
    return tiny


def run_graph(nodes):
    spans: dict[str, dict] = {}
    spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": {"dataflow": 2, "nodes": nodes, "edges": []}})
    ProtocolExecutor(on_node_span=lambda nid, span: spans.setdefault(nid, span)).run(spec)
    return spans


RECORDS = [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"}]


def test_every_span_carries_every_field_and_the_load_once(tiny_hub):
    spans = run_graph([
        {"id": "cap", "block": "activations/capture",
         "params": {"model": MODEL, "layers": [0, 2]}, "inputs": {"records": RECORDS}},
        {"id": "gen", "block": "text/generate",
         "params": {"model": MODEL, "n": 2, "max_tokens": 3, "temperature": 0.0},
         "inputs": {"records": RECORDS}},
    ])
    assert list(spans) == ["cap", "gen"]
    for span in spans.values():
        assert tuple(span) == SPAN_FIELDS
        assert span["ambient"] is None and span["quiet"] is None
        assert span["compute_seconds"] >= 0 and span["peak_memory_bytes"] > 0
    cap, gen = spans["cap"], spans["gen"]
    assert cap["model_load_seconds"] > 0 and gen["model_load_seconds"] is None
    assert cap["forwards"] == len(RECORDS) and cap["backwards"] == 0
    assert cap["bytes_captured"] > 0 and cap["tokens_out"] == 0
    assert gen["bytes_captured"] == 0 and gen["backwards"] == 0
    assert gen["tokens_in"] > 0 and 0 < gen["tokens_out"] <= 2 * len(RECORDS) * 3
    assert gen["forwards"] >= len(RECORDS)


def test_both_capture_ops_count_the_tokens_they_read_and_the_bytes_they_capture(tiny_hub):
    spans = run_graph([
        {"id": "cap", "block": "activations/capture",
         "params": {"model": MODEL, "layers": [0, 2], "position": "last"},
         "inputs": {"records": RECORDS}},
        {"id": "tok", "block": "activations/capture-tokens",
         "params": {"model": MODEL, "layers": [0, 2], "storage": "tensor"},
         "inputs": {"records": RECORDS}},
    ])
    for span in spans.values():
        assert span["forwards"] == len(RECORDS)
        assert span["tokens_in"] >= len(RECORDS) * 3
        assert span["bytes_captured"] > 0 and span["tokens_out"] == 0
    assert spans["cap"]["tokens_in"] == spans["tok"]["tokens_in"]


def test_a_nested_span_folds_into_its_parent():
    with open_span() as outer:
        add_to_span(forwards=1)
        with open_span() as inner:
            add_to_span(forwards=2, tokens_out=5, model_load_seconds=0.5)
        add_to_span(backwards=1)
    assert inner.counts["forwards"] == 2
    assert outer.counts["forwards"] == 3 and outer.counts["tokens_out"] == 5
    assert outer.counts["backwards"] == 1 and outer.model_load_seconds == 0.5
    add_to_span(forwards=1)
    assert outer.counts["forwards"] == 3
