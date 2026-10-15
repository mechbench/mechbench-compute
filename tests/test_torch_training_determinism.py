from __future__ import annotations

import importlib.util
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("nnsight is not installed: pip install 'mechbench-compute[torch]'",
                allow_module_level=True)
torch = pytest.importorskip("torch")

from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.torch_backend import determinism
from tests.test_torch_training import ANCHORS, DECISION, RECORDS
from tests.tiny_torch_models import build_tiny_model

PARAMS = {**DECISION, "checkpoint_every": 2}

OPERATOR = {"model": "tiny", "target": {"uniform": ["mat", "dog", "cat"]}, "closer": " ran",
            "steps": 8, "lr": 0.05, "seed": 3, "checkpoint_every": 2, "batch": {"target": 3},
            "operator": {"layers": [1], "function": "mlp", "width": 4, "gate": True,
                         "penalty": {"l1": 0.01}}}


def train(model, params=PARAMS, **ctx):
    states: list[dict] = []
    out = train_op.run(Context(loaded=model, on_checkpoint=states.append, **ctx),
                       {"records": RECORDS, "anchors": ANCHORS}, params)
    kept = {i["id"]: i for i in (out.get("checkpoints") or {}).get("items", [])}
    return SimpleNamespace(out=out["out"], kept=kept, states=states)


def through_npz(state: dict) -> dict:
    def save_load(arrays: dict) -> dict:
        buffer = io.BytesIO()
        np.savez(buffer, **{k: np.asarray(v) for k, v in arrays.items()})
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    opt = state["opt_state"]
    arrays = {k: v for k, v in opt.items() if hasattr(v, "shape")}
    return {"step": int(state["step"]), "weights": save_load(state["weights"]),
            "opt_state": {**save_load(arrays), **{k: v for k, v in opt.items() if k not in arrays}},
            "np_rng": json.loads(json.dumps(state["np_rng"])), "mx_key": state["mx_key"]}


def assert_same_state(a: dict, b: dict) -> None:
    assert a["step"] == b["step"] and a["np_rng"] == b["np_rng"]
    for part in ("weights", "opt_state"):
        assert set(a[part]) == set(b[part]), part
        for k in a[part]:
            x, y = np.asarray(a[part][k]), np.asarray(b[part][k])
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (part, k)


@pytest.mark.parametrize("params", [PARAMS, OPERATOR], ids=["lora", "operator"])
def test_two_seeded_runs_write_the_same_weights_and_optimizer_state(params):
    model = build_tiny_model("gemma3")
    a, b = train(model, params), train(model, params)
    assert [s["step"] for s in a.states] == [2, 4, 6]
    for x, y in zip(a.states, b.states, strict=True):
        assert_same_state(x, y)
    if "operator" in params:
        assert a.out["parameters"] == b.out["parameters"]
    else:
        assert a.out["data"] == b.out["data"]
        assert [i["data"] for i in a.kept.values()] == [i["data"] for i in b.kept.values()]
    numerics = a.out["train"]["numerics"]
    assert numerics["deterministic"] is True and numerics["deterministic_algorithms"] is True
    assert numerics["deterministic_warn_only"] is False and "note" not in numerics


@pytest.mark.parametrize("params", [PARAMS, OPERATOR], ids=["lora", "operator"])
def test_a_run_resumed_from_a_checkpoint_saved_through_npz_ends_on_the_uninterrupted_bytes(params):
    model = build_tiny_model("llama")
    whole = train(model, params)
    saved = through_npz(whole.states[1])
    assert_same_state(saved, whole.states[1])
    kept = {k: v for k, v in whole.kept.items() if v["coords"]["step"] <= saved["step"]}
    resumed = train(model, params, resume_state=saved, resume_items=kept)
    assert [s["step"] for s in resumed.states] == [6]
    assert_same_state(resumed.states[0], whole.states[2])
    if "operator" in params:
        assert resumed.out["parameters"] == whole.out["parameters"]
    else:
        assert resumed.out["data"] == whole.out["data"]


def test_the_mode_holds_torch_to_deterministic_algorithms_while_training_and_puts_it_back():
    model = build_tiny_model("qwen2")
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    seen: list[tuple[bool, bool]] = []

    def on_item(*_):
        seen.append((torch.are_deterministic_algorithms_enabled(),
                     torch.is_deterministic_algorithms_warn_only_enabled()))

    train(model, {**PARAMS, "keep_checkpoints": False}, on_item=on_item)
    assert seen and set(seen) == {(True, False)}
    seen.clear()
    fast = train(model, {**PARAMS, "keep_checkpoints": False, "deterministic": False}, on_item=on_item)
    assert set(seen) == {before}
    assert (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled()) == before
    numerics = fast.out["train"]["numerics"]
    assert numerics["deterministic"] is False and "two runs with the same seed" in numerics["note"]


def test_on_mlx_the_adapter_header_carries_no_numerics():
    pytest.importorskip("mlx.core")
    from tests.mlx_twins import build_on_mlx

    on_mlx = build_on_mlx(build_tiny_model("llama"))
    assert "numerics" not in train(on_mlx).out["train"]


def test_on_cuda_the_mode_needs_a_deterministic_cublas_workspace(monkeypatch):
    gpu = SimpleNamespace(type="cuda")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        determinism.check_workspace(gpu)
    for held in determinism.DETERMINISTIC_WORKSPACES:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", held)
        determinism.check_workspace(gpu)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    determinism.check_workspace(SimpleNamespace(type="cpu"))
    numerics = determinism.read_training_numerics(gpu, True)
    assert numerics["attention_backward"] == "math" and numerics["cublas_workspace"] == ":0:0"
    assert determinism.read_training_numerics(gpu, False)["attention_backward"] == "default"
