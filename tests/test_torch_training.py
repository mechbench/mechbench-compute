from __future__ import annotations

import importlib.metadata
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("nnsight is not installed: pip install 'mechbench-compute[torch]'",
                allow_module_level=True)
torch = pytest.importorskip("torch")

from safetensors.torch import load as load_tensors

from mechbench_compute import backends
from mechbench_compute.adapter_keys import ADAPTER_KEYS
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.torch_backend import checkpointing
from mechbench_compute.torch_backend.lora import (
    fuse_adapter_stack,
    restore_adapter_stack,
)
from mechbench_compute.torch_backend.train_timing import time_training
from tests.tiny_torch_models import KIT_MODELS, build_tiny_model

MODELS = tuple(name for name, _ in KIT_MODELS)

RECORDS = [{"id": "a", "user": "the cat sat on a"}, {"id": "b", "user": "the dog ran on a"}]

ANCHORS = [{"id": "x", "user": "the cat sat on the", "answer": "mat"}]

DECISION = {"model": "tiny", "target": {"uniform": ["mat", "dog", "cat"]}, "closer": " ran",
            "steps": 8, "lr": 0.05, "seed": 3, "lora": {"rank": 2, "alpha": 4},
            "checkpoint_every": 4, "keep_checkpoints": True}

IDS = [2, 7, 8, 9, 10, 11]


def train(model, params=DECISION, **ctx):
    states: list[dict] = []
    out = train_op.run(Context(loaded=model, on_checkpoint=states.append, **ctx),
                       {"records": RECORDS, "anchors": ANCHORS}, params)
    return SimpleNamespace(adapter=out["out"], kept=(out.get("checkpoints") or {}).get("items", []), states=states)


def read_target_loss(model, adapter=None) -> float:
    from mechbench_compute.distill import encode, render

    tok = model.tokenizer
    targets = [encode(tok, w)[0] for w in DECISION["target"]["uniform"]]
    handles = fuse_adapter_stack(model.lm, [adapter]) if adapter is not None else []
    try:
        total = 0.0
        for record in RECORDS:
            row = model.run(model.make_ids(render(model, record).ids)).logits[0, -1].float()
            total += float(torch.logsumexp(row, -1) - row[targets].mean())
    finally:
        restore_adapter_stack(model.lm, handles)
    return total / len(RECORDS)


def read_logits(model) -> np.ndarray:
    return model.run(model.make_ids(IDS)).logits.detach().float().cpu().numpy()


@pytest.mark.parametrize("name", MODELS)
def test_the_decision_objective_trains_on_torch_its_loss_falls_and_the_base_comes_back(name):
    model = build_tiny_model(name)
    before = {k: (p.detach().clone(), p.requires_grad) for k, p in model._model.named_parameters()}
    run = train(model)
    assert read_target_loss(model, run.adapter) < read_target_loss(model) - 0.1
    assert run.adapter["train"]["final_loss"] == round(run.kept[-1]["loss"], 4)
    assert run.kept[-1]["data"] == run.adapter["data"]
    weights = load_tensors(run.adapter["data"])
    assert all(ADAPTER_KEYS.key_re.match(k) for k in weights)
    assert {k.rsplit(".", 2)[-2] for k in weights} == {"q_proj", "v_proj"}
    assert all(w.dtype == torch.float32 for w in weights.values())
    assert run.adapter["lora"]["params"] == sum(w.numel() for w in weights.values())
    for k, p in model._model.named_parameters():
        assert torch.equal(p, before[k][0]) and p.requires_grad == before[k][1], k


def test_two_seeded_runs_write_the_same_bytes_and_checkpointing_changes_none(monkeypatch):
    model = build_tiny_model("gemma3")
    a, b = train(model), train(model)
    assert a.adapter["data"] == b.adapter["data"]
    assert [i["data"] for i in a.kept] == [i["data"] for i in b.kept]
    other = train(model, {**DECISION, "seed": 4})
    assert other.adapter["data"] != a.adapter["data"]
    monkeypatch.setenv(checkpointing.SWITCH, "on")
    assert train(model).adapter["data"] == a.adapter["data"]


def test_a_resumed_run_ends_where_the_uninterrupted_one_did():
    model = build_tiny_model("llama")
    whole = train(model)
    assert [s["step"] for s in whole.states] == [4]
    kept = {i["id"]: i for i in whole.kept if i["coords"]["step"] <= 4}
    resumed = train(model, resume_state=whole.states[0], resume_items=kept)
    assert resumed.adapter["data"] == whole.adapter["data"]
    assert [i["data"] for i in resumed.kept] == [i["data"] for i in whole.kept]


def test_the_trained_adapter_fuses_on_torch_and_moves_the_logits():
    model = build_tiny_model("qwen2")
    run = train(model)
    base = read_logits(model)
    handles = fuse_adapter_stack(model.lm, [run.adapter])
    try:
        fused = read_logits(model)
    finally:
        restore_adapter_stack(model.lm, handles)
    assert np.abs(fused - base).max() > 1e-3
    assert np.array_equal(read_logits(model), base)


def test_an_operator_is_refused_on_torch_by_name():
    model = build_tiny_model("gemma3")
    with pytest.raises(ValueError, match="operator trains on the mlx backend only"):
        train(model, {**DECISION, "keep_checkpoints": False,
                      "operator": {"layers": [1], "function": "affine"}})


def test_checkpointing_is_decided_by_the_switch_and_by_the_free_memory(monkeypatch):
    model = build_tiny_model("gemma3")
    assert checkpointing.decide_checkpointing(model, 10_000) is False
    monkeypatch.setattr(checkpointing, "read_free_bytes", lambda device: 1 << 20)
    assert checkpointing.decide_checkpointing(model, 10_000) is True
    assert checkpointing.decide_checkpointing(model, 1) is False
    monkeypatch.setenv(checkpointing.SWITCH, "off")
    assert checkpointing.decide_checkpointing(model, 10_000) is False
    monkeypatch.setenv(checkpointing.SWITCH, "sometimes")
    with pytest.raises(ValueError, match="on, off or auto"):
        checkpointing.decide_checkpointing(model, 1)


def test_the_timing_helper_reports_and_leaves_the_model_as_it_was():
    model = build_tiny_model("gemma4")
    base = read_logits(model)
    out = time_training(model, steps=2, items=2, prompt_tokens=8, project=(60, 240))
    assert out["steps"] == 2 and out["seconds_per_step"] > 0 and out["tokens_per_s"] > 0
    assert out["tokens"] == 2 * 2 * 8
    assert set(out["projected_seconds"]) == {60, 240}
    assert out["peak_memory_bytes"] is None or out["peak_memory_bytes"] > 0
    assert np.array_equal(read_logits(model), base)


@pytest.fixture
def torch_job(monkeypatch):
    from mechbench_compute.torch_backend.model import TorchModel

    tiny = build_tiny_model("gemma3")
    monkeypatch.setattr(TorchModel, "load", classmethod(lambda cls, model_id, **_: tiny))
    return tiny


def test_a_torch_training_job_reports_its_resources(torch_job):
    params = {**DECISION, "model": "tiny/gemma3@rev", "keep_checkpoints": False}
    graph = {"dataflow": 2, "edges": [], "nodes": [
        {"id": "train", "block": "adapter/train", "params": params,
         "inputs": {"records": RECORDS, "anchors": ANCHORS}}]}
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
        "graph": graph, "requirements": {"backend": "torch"}}))
    payload = out.payload
    assert payload["outputs"]["train"]["train"]["steps"] == 8
    resources = payload["resources"]
    assert resources["hardware"]["backend"] == "torch"
    assert resources["hardware"]["torch"] == importlib.metadata.version("torch")
    training = resources["training"]["train"]
    assert training["backend"] == "torch" and training["steps"] == 8 and training["from_step"] == 0
    assert training["tokens"] > 0 and training["tokens_per_s"] > 0
    assert training["gradient_checkpointing"] is False
    assert "adapter/train" in next(b for b in backends.BACKENDS if b.name == "torch").ops
