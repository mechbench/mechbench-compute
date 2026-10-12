from __future__ import annotations

import gc
import importlib.util
import weakref

import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("nnsight is not installed: pip install 'mechbench-compute[torch]'",
                allow_module_level=True)
torch = pytest.importorskip("torch")

from mechbench_compute import bench
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.torch_backend import tracing
from mechbench_compute.torch_backend.model import TorchModel
from tests.tiny_torch_models import build_tiny_model

RECORDS = [{"id": "a", "user": "the cat sat on a"}, {"id": "b", "user": "the dog ran on a"}]

ANCHORS = [{"id": "x", "user": "the cat sat on the", "answer": "mat"}]

TRAIN = {"model": "tiny/a@rev", "target": {"uniform": ["mat", "dog", "cat"]}, "closer": " ran",
         "steps": 4, "lr": 0.05, "seed": 3, "lora": {"rank": 2, "alpha": 4},
         "checkpoint_every": 2, "keep_checkpoints": True}

KEPT = "you/lab/results/j_train/train"

ORGANISM = {"base": {"hf": "tiny/a@rev"}, "adapters": [{"bench": KEPT}]}


class Loads:
    def __init__(self) -> None:
        self.held: list[weakref.ref] = []
        self.alive_at_load: list[tuple[str, int]] = []

    def record(self, model_id: str, model: TorchModel) -> TorchModel:
        self.held += [weakref.ref(model), weakref.ref(model._model),
                      weakref.ref(next(model._model.parameters()))]
        return model

    def count_alive(self) -> int:
        return sum(ref() is not None for ref in self.held)


@pytest.fixture
def loads(tmp_path, monkeypatch):
    dirs = {}
    for name in ("a", "b"):
        tiny = build_tiny_model("gemma3")
        tiny._model.save_pretrained(tmp_path / name)
        tiny.tokenizer.save_pretrained(tmp_path / name)
        dirs[f"tiny/{name}@rev"] = tmp_path / name
    kept = train_op.run(Context(loaded=build_tiny_model("gemma3")),
                        {"records": RECORDS, "anchors": ANCHORS}, TRAIN)["checkpoints"]["items"][0]
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id.split("@")[0], "rev", dirs[model_id]))
    monkeypatch.setattr(bench, "fetch", lambda label, with_meta=False, **_:
                        (kept, {"content_hash": "kept"}) if with_meta else kept)
    seen = Loads()
    load = TorchModel.load.__func__

    def watched(cls, model_id, **kwargs):
        seen.alive_at_load.append((model_id, seen.count_alive()))
        return seen.record(model_id, load(cls, model_id, **kwargs))

    monkeypatch.setattr(TorchModel, "load", classmethod(watched))
    return seen


def capture(nid, model):
    return {"id": nid, "block": "activations/capture",
            "inputs": {"records": RECORDS},
            "params": {"model": model, "layers": [1], "point": "resid_post"}}


def generate(nid, model):
    return {"id": nid, "block": "text/generate",
            "inputs": {"records": RECORDS}, "params": {"model": model, "n": 1, "max_tokens": 3}}


def run_job(executor, nodes):
    out = executor.run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
        "graph": {"dataflow": 2, "nodes": nodes, "edges": []},
        "requirements": {"class": "local", "backend": "torch"}}))
    payload = out.payload
    assert not payload.get("missing"), payload.get("missing")
    return payload


def test_an_organism_traced_and_then_switched_away_from_leaves_no_weights_behind(loads):
    executor = ProtocolExecutor()
    run_job(executor, [capture("read", ORGANISM), generate("stories", ORGANISM)])
    run_job(executor, [generate("stories", "tiny/b@rev")])
    run_job(executor, [generate("stories", ORGANISM), capture("read", ORGANISM)])
    assert [m for m, _ in loads.alive_at_load] == ["tiny/a@rev", "tiny/b@rev", "tiny/a@rev"]
    assert [alive for _, alive in loads.alive_at_load] == [0, 0, 0]


def test_a_switch_between_two_plain_models_in_one_job_holds_one_at_a_time(loads):
    run_job(ProtocolExecutor(), [capture("a", "tiny/a@rev"), capture("b", "tiny/b@rev"),
                                 generate("organism", ORGANISM)])
    assert [m for m, _ in loads.alive_at_load] == ["tiny/a@rev", "tiny/b@rev", "tiny/a@rev"]
    assert [alive for _, alive in loads.alive_at_load] == [0, 0, 0]


def trace_once() -> weakref.ref:
    module = torch.nn.Linear(2, 2)
    envoy = tracing.wrap_model(module)
    assert tracing.wrap_model(module) is envoy
    with envoy.trace(torch.ones(1, 2)):
        envoy.output
    return weakref.ref(module)


def test_a_traced_module_is_freed_with_its_envoy():
    held = trace_once()
    gc.collect()
    assert held() is None
