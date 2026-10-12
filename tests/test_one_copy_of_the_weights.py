from __future__ import annotations

import dataclasses
import weakref

import pytest

from mechbench_compute import architectures, bench
from mechbench_compute import model as model_mod
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from tests.tiny_models import build_tiny_model

RECORDS = [{"id": "a", "user": "the cat sat on a"}, {"id": "b", "user": "the dog ran on a"}]

ANCHORS = [{"id": "x", "user": "the cat sat on the", "answer": "mat"}]

TRAIN = {"model": "tiny/a@rev", "target": {"uniform": ["mat", "dog", "cat"]}, "closer": " ran",
         "steps": 4, "lr": 0.05, "seed": 3, "lora": {"rank": 2, "alpha": 4},
         "checkpoint_every": 2, "keep_checkpoints": True}

KEPT = "you/lab/results/j_train/train"

ORGANISM = {"base": {"hf": "tiny/a@rev"}, "adapters": [{"bench": KEPT}]}


@pytest.fixture
def loads(tmp_path, monkeypatch):
    kept = train_op.run(Context(loaded=build_tiny_model("gemma3")),
                        {"records": RECORDS, "anchors": ANCHORS}, TRAIN)["checkpoints"]["items"][0]
    held: list[weakref.ref] = []
    alive_at_load: list[tuple[str, int]] = []

    def build(_path):
        tiny = build_tiny_model("gemma3")
        return tiny._model, tiny._processor

    architecture = dataclasses.replace(architectures.BY_MODEL_TYPE["gemma3"], load=build)
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": "gemma3"})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)
    monkeypatch.setattr(bench, "fetch", lambda label, with_meta=False, **_:
                        (kept, {"content_hash": "kept"}) if with_meta else kept)
    load = model_mod.Model.load.__func__

    def watched(cls, model_id, **kwargs):
        alive_at_load.append((model_id, sum(ref() is not None for ref in held)))
        model = load(cls, model_id, **kwargs)
        held.extend((weakref.ref(model), weakref.ref(model.lm)))
        return model

    monkeypatch.setattr(model_mod.Model, "load", classmethod(watched))
    return alive_at_load


def capture(nid, model):
    return {"id": nid, "block": "activations/capture", "inputs": {"records": RECORDS},
            "params": {"model": model, "layers": [1], "point": "resid_post"}}


def generate(nid, model):
    return {"id": nid, "block": "text/generate",
            "inputs": {"records": RECORDS}, "params": {"model": model, "n": 1, "max_tokens": 3}}


def run_job(executor, nodes):
    out = executor.run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
        "graph": {"dataflow": 2, "nodes": nodes, "edges": []}}))
    payload = out.payload
    assert not payload.get("missing"), payload.get("missing")
    return payload


def test_an_organism_switched_away_from_and_back_leaves_no_weights_behind(loads):
    executor = ProtocolExecutor()
    run_job(executor, [capture("read", ORGANISM), generate("stories", ORGANISM)])
    run_job(executor, [generate("stories", "tiny/b@rev")])
    run_job(executor, [generate("stories", ORGANISM), capture("read", ORGANISM)])
    assert loads == [("tiny/a@rev", 0), ("tiny/b@rev", 0), ("tiny/a@rev", 0)]


def test_a_switch_between_two_plain_models_in_one_job_holds_one_at_a_time(loads):
    run_job(ProtocolExecutor(), [capture("a", "tiny/a@rev"), capture("b", "tiny/b@rev"),
                                 generate("organism", ORGANISM)])
    assert loads == [("tiny/a@rev", 0), ("tiny/b@rev", 0), ("tiny/a@rev", 0)]
