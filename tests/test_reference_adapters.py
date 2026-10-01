from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import generate as generate_mod
from mechbench_compute import lora
from mechbench_compute.distill import render
from mechbench_compute.lexicon._base import In, Op
from mechbench_compute.model_ref import ModelRef
from mechbench_compute.ops import Context
from mechbench_compute.ops.text import resample as resample_op
from mechbench_compute.protocol import ProtocolExecutor
from tests.test_circuits import build_adapter
from tests.tiny_models import build_tiny_model

RECORD = {"id": "r", "user": "the cat sat on a mat", "outcomes": ["cat", "dog", "mat"]}


def read_logits(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    return {"logits": np.array(model.run(render(model, RECORD).array).logits[0, -1]).tolist()}


def build_op(*, port: bool):
    inputs = (In("adapter", "adapter/lora", "An adapter.", required=False),) if port else ()
    declared = Op(name="probe/logits", summary="Reads logits.", description="Reads logits.",
                  params=(), inputs=inputs, needs=frozenset({"model.forward"}))
    return SimpleNamespace(op=declared, module=SimpleNamespace(run=read_logits), scope=None)


def build_payload(model, tmp_path, name, sign):
    path = tmp_path / f"{name}.safetensors"
    weights = build_adapter(model)
    mx.save_safetensors(str(path), {k: v * sign if k.endswith("lora_b") else v for k, v in weights.items()})
    return {"data": path.read_bytes(), "lora": {"rank": 2, "alpha": 2}}


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


@pytest.fixture
def executor(tiny):
    ex = ProtocolExecutor()
    ex._model, ex._model_id = tiny, "tiny"
    return ex


def adapted_ref(*payloads):
    labels = tuple(f"you/lab/adapter-{i}" for i in range(len(payloads)))
    return ModelRef(base_kind="hf", base="tiny", adapter_labels=labels, adapter_payloads=payloads)


def logits_under(model, payloads):
    handles = lora.fuse_adapter_stack(model.lm, list(payloads), keys=model.architecture.adapter_keys)
    try:
        return read_logits(Context(loaded=model), {}, {})["logits"]
    finally:
        lora.restore_adapter_stack(model.lm, handles)


class TestTheReferenceFusesForEveryModelOp:
    def test_an_op_without_the_port_reads_the_adapted_model(self, executor, tiny, tmp_path):
        payload = build_payload(tiny, tmp_path, "die", 1)
        bare = read_logits(Context(loaded=tiny), {}, {})["logits"]
        out = executor._run_op(build_op(port=False), {}, {"model": adapted_ref(payload)})
        assert out["logits"] == pytest.approx(logits_under(tiny, [payload]), abs=1e-5)
        assert out["logits"] != pytest.approx(bare, abs=1e-3)
        assert out["fused"] == [{"bench": "you/lab/adapter-0"}]

    def test_the_model_is_bare_again_after_the_op(self, executor, tiny, tmp_path):
        payload = build_payload(tiny, tmp_path, "die", 1)
        bare = read_logits(Context(loaded=tiny), {}, {})["logits"]
        executor._run_op(build_op(port=False), {}, {"model": adapted_ref(payload)})
        assert read_logits(Context(loaded=tiny), {}, {})["logits"] == bare
        assert tiny.fused_reference is None

    def test_with_the_port_both_fuse_the_reference_first(self, executor, tiny, tmp_path):
        own, extra = build_payload(tiny, tmp_path, "own", 1), build_payload(tiny, tmp_path, "extra", -0.5)
        out = executor._run_op(build_op(port=True), {"adapter": extra},
                               {"model": adapted_ref(own), "adapter_scale": 0.5})
        assert out["fused"] == [{"bench": "you/lab/adapter-0"}, {"port": "adapter", "scale": 0.5}]
        handles = lora.fuse_adapter_stack(tiny.lm, [own, extra], 0.5, keys=tiny.architecture.adapter_keys)
        try:
            expected = read_logits(Context(loaded=tiny), {}, {})["logits"]
        finally:
            lora.restore_adapter_stack(tiny.lm, handles)
        assert out["logits"] == pytest.approx(expected, abs=1e-5)

    def test_an_unresolved_reference_is_refused_by_name(self, executor):
        ref = ModelRef(base_kind="hf", base="tiny", adapter_labels=("you/lab/die",))
        with pytest.raises(ValueError, match=r"probe/logits: the model reference hf:tiny \(\+1 adapter\) "
                                             r"carries the adapters you/lab/die"):
            executor._run_op(build_op(port=False), {}, {"model": ref})


class TestResampleTakesTheAdapterPort:
    @pytest.fixture(autouse=True)
    def no_stop(self, monkeypatch):
        monkeypatch.setattr(generate_mod, "_stop_ids", lambda tokenizer: set())

    PARAMS = {"k": 2, "max_tokens": 3, "seed": 5, "cue": " the", "where": "position"}

    def test_the_port_and_the_reference_fuse_and_the_header_names_both(self, executor, tiny, tmp_path):
        own, extra = build_payload(tiny, tmp_path, "own", 1), build_payload(tiny, tmp_path, "extra", -0.5)
        ref = adapted_ref(own)
        out = executor._run_op("text/resample", {"records": [RECORD], "adapter": extra},
                               {**self.PARAMS, "model": ref})
        handles = lora.fuse_adapter_stack(tiny.lm, [own, extra], keys=tiny.architecture.adapter_keys)
        try:
            expected = resample_op.run(Context(loaded=tiny), {"records": [RECORD]},
                                       {**self.PARAMS, "model": ref})
        finally:
            lora.restore_adapter_stack(tiny.lm, handles)
        assert out["out"]["items"] == expected["out"]["items"]
        assert out["out"]["fused"] == [{"bench": "you/lab/adapter-0"}, {"port": "adapter"}]
        assert out["out"]["model"]["adapters"] == [{"bench": "you/lab/adapter-0"}]

    def test_the_reference_alone_is_not_the_bare_model(self, executor, tiny, tmp_path):
        own = build_payload(tiny, tmp_path, "own", 1)
        adapted = executor._run_op("text/resample", {"records": [RECORD]},
                                   {**self.PARAMS, "model": adapted_ref(own)})
        bare = executor._run_op("text/resample", {"records": [RECORD]}, {**self.PARAMS, "model": "tiny"})
        assert adapted["out"]["fused"] == [{"bench": "you/lab/adapter-0"}]
        assert "fused" not in bare["out"]
        assert adapted["out"]["items"] != bare["out"]["items"]
