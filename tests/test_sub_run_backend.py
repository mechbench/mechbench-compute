from __future__ import annotations

import contextlib

import pytest

from mechbench_compute import backends
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

MODEL = "tiny/gemma3@rev"

READ = {"id": "read", "block": "logits/read",
        "params": {"model": {"$param": "model"}, "top_k": 3},
        "inputs": {"conditions": [{"id": "c", "prompt": "the cat sat on the", "template": "raw"}]}}


class Loaded(Exception):
    pass


@pytest.fixture
def loads(monkeypatch):
    seen: list[tuple[str, str | None]] = []

    def model_loaded(self, model_id):
        seen.append((self._backend.name, self._accelerator))
        raise Loaded

    monkeypatch.setattr(ProtocolExecutor, "_model_loaded", model_loaded)
    return seen


def run_job(graph: dict, requirements: dict) -> None:
    spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": {"dataflow": 2, **graph}, "params": {"model": MODEL},
                               "requirements": requirements})
    with contextlib.suppress(Exception):
        ProtocolExecutor().run(spec)


def test_a_map_body_loads_its_model_on_the_job_s_backend_and_accelerator(loads):
    graph = {"nodes": [{"id": "each", "block": "records/map",
                        "params": {"over": [1], "body": {"nodes": [READ], "edges": []}}}],
             "edges": []}
    run_job(graph, {"class": "local", "backend": "torch", "accelerator": "cuda"})
    assert loads == [("torch", "cuda")]


def test_a_fold_body_loads_its_model_on_the_job_s_backend_and_accelerator(loads):
    graph = {"nodes": [{"id": "each", "block": "records/fold",
                        "params": {"steps": 1, "body": {"nodes": [READ], "edges": []}},
                        "inputs": {"state": [{"id": "s"}]}}],
             "edges": []}
    run_job(graph, {"class": "local", "backend": "torch", "accelerator": "cuda"})
    assert loads == [("torch", "cuda")]


def test_a_sub_run_of_one_operation_keeps_the_parent_s_backend_and_accelerator(loads):
    parent = ProtocolExecutor()
    parent._backend, parent._accelerator = backends.find("torch"), "cuda"
    with pytest.raises(Loaded):
        parent.run_sub("logits/read", READ["inputs"], {"model": MODEL, "top_k": 3})
    assert loads == [("torch", "cuda")]
